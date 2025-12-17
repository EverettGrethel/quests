import os
import json
import fcntl
import time
from contextlib import contextmanager
import subprocess
import atexit
import signal


def parse_cuda_device_index(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":")[1])
    raise ValueError(f"Unsupported device format: {device}")


def query_free_mem_gb_nvidia_smi(dev_idx: int) -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", str(dev_idx),
             "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        return float(out.strip()) / 1024.0
    except:
        return 0.0


def query_total_mem_gb_nvidia_smi(dev_idx: int) -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", str(dev_idx),
             "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        return float(out.strip()) / 1024.0
    except:
        return 0.0

def _reservation_file(dev_idx: int) -> str:
    return f"/tmp/entropy_gpu_{dev_idx}.json"

def _read_reservation_from_fd(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    raw = os.read(fd, 64_000).decode("utf-8")  # large enough buffer
    if not raw:
        return {}  # pid (str) -> gb
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        # Corrupted file — reset to empty
        return {}

def _write_reservation_to_fd(fd, mapping: dict):
    os.lseek(fd, 0, os.SEEK_SET)
    encoded = json.dumps(mapping).encode("utf-8")
    os.write(fd, encoded)
    pos = os.lseek(fd, 0, os.SEEK_CUR)
    os.ftruncate(fd, pos)

def _cleanup_dead_pids(mapping: dict) -> dict:
    """Remove PIDs that are no longer running."""
    alive = {}
    for pid_str, gb in mapping.items():
        try:
            pid = int(pid_str)
        except Exception:
            continue
        # kill 0 checks existence without sending a signal
        try:
            os.kill(pid, 0)
            alive[pid_str] = gb
        except ProcessLookupError:
            # dead, drop it
            continue
        except PermissionError:
            # we can't signal it but assume it's alive
            alive[pid_str] = gb
    return alive

# ---------- atomic reservation attempt ----------

def _try_reserve(dev_idx: int, pid: int, amount_gb: float) -> bool:
    """
    Acquire file lock, remove stale PIDs, re-query free mem, and if usable >= amount_gb
    add (or increase) the pid entry and write back. Return True if reserved.
    """
    path = _reservation_file(dev_idx)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            mapping = _read_reservation_from_fd(fd)     # pid_str -> float
            mapping = _cleanup_dead_pids(mapping)

            # sum of reserved by alive holders
            reserved_total = sum(float(v) for v in mapping.values())

            # re-query free memory while we still hold lock (reduces race window)
            free_gb = query_free_mem_gb_nvidia_smi(dev_idx)
            usable_gb = free_gb - reserved_total

            if usable_gb >= amount_gb:
                # grant reservation: add or increment this pid
                mapping[str(pid)] = float(mapping.get(str(pid), 0.0)) + float(amount_gb)
                # write mapping back
                _write_reservation_to_fd(fd, mapping)
                return True
            else:
                # not enough usable memory
                return False
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)

def _release_pid_reservation(dev_idx: int, pid: int, amount_gb: float = None):
    """
    Atomically release reservation for pid.
    If amount_gb is None, remove pid entry entirely.
    Otherwise decrement by amount_gb and drop if <= 0.
    """
    path = _reservation_file(dev_idx)
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    except FileNotFoundError:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            mapping = _read_reservation_from_fd(fd)
            if str(pid) not in mapping:
                return
            if amount_gb is None:
                mapping.pop(str(pid), None)
            else:
                remaining = float(mapping.get(str(pid), 0.0)) - float(amount_gb)
                if remaining <= 0:
                    mapping.pop(str(pid), None)
                else:
                    mapping[str(pid)] = remaining
            _write_reservation_to_fd(fd, mapping)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)

# ---------- improved exclusive_gpu using above ----------

@contextmanager
def exclusive_gpu(devices, min_free_gb: float = 10.0, poll_s: float = 30.0):
    """
    Multi-process friendly reservation context:
      exclusive_gpu("cuda:0")
      exclusive_gpu(["cuda:0", "cuda:1"])

    Performs atomic reservation checks and cleans stale PID entries.
    Registers best-effort release handlers for atexit/signals.
    """
    if isinstance(devices, str):
        device_list = [devices]
    else:
        device_list = list(devices)

    pid = os.getpid()
    selected_device = None
    selected_dev_idx = None
    reserved_amount = float(min_free_gb)

    # helper to release for our pid (used in finally and signal handlers)
    def _release_all():
        nonlocal selected_dev_idx
        if selected_dev_idx is not None:
            try:
                _release_pid_reservation(selected_dev_idx, pid)
            except Exception:
                pass

    # register best-effort cleanup
    atexit.register(_release_all)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        try:
            signal.signal(sig, lambda *_args, **_kwargs: _release_all())
        except Exception:
            # some signals can't be handled on some platforms — ignore
            pass

    try:
        while selected_device is None:
            for dev in device_list:
                if not isinstance(dev, str) or not dev.startswith("cuda:"):
                    raise ValueError(f"Invalid device: {dev}")

                dev_idx = parse_cuda_device_index(dev)

                # Attempt atomic reservation: this will clean stale entries, re-check free memory
                ok = _try_reserve(dev_idx, pid, reserved_amount)
                if ok:
                    selected_device = dev
                    selected_dev_idx = dev_idx
                    free_gb = query_free_mem_gb_nvidia_smi(dev_idx)
                    # optional: read how much the file says we reserved
                    print(f"[{dev}] Reservation granted: {reserved_amount:.2f} GB (free={free_gb:.2f} GiB)")
                    break
                else:
                    # read debug info (best-effort)
                    reserved_snapshot = _atomic_read_snapshot(dev_idx=dev_idx)
                    free_gb = query_free_mem_gb_nvidia_smi(dev_idx)
                    usable = free_gb - reserved_snapshot
                    print(f"[{dev}] Insufficient usable mem: free={free_gb:.2f}, reserved={reserved_snapshot:.2f}, usable={usable:.2f}, need={reserved_amount:.2f}")

            if selected_device is None:
                time.sleep(poll_s)

        try:
            yield selected_device

        finally:
            # release our reservation
            _release_pid_reservation(selected_dev_idx, pid)
    finally:
        # attempt final cleanup just in case
        _release_all()

# ---------- small helper to read total reserved snapshot for logging ----------

def _atomic_read_snapshot(dev_idx: int) -> float:
    """Return the sum of currently reserved GBs (cleans dead PIDs)."""
    path = _reservation_file(dev_idx)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            mapping = _read_reservation_from_fd(fd)
            mapping = _cleanup_dead_pids(mapping)
            total = sum(float(v) for v in mapping.values())
            return total
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)