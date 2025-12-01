import os
import fcntl
import time
from contextlib import contextmanager
import argparse
import json
import subprocess
from joblib import Parallel, delayed
import numpy as np
import torch

from ase.io import read
from pyace import create_multispecies_basis_config
from pyace.activelearning import compute_B_projections
from quests.gpu.entropy import entropy


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--elements",          required=True, nargs="+", help="Element symbols, e.g. C or C Au")
    parser.add_argument("--deltaSplineBins",   required=True, type=str)
    parser.add_argument("--npot",              required=True, type=str)
    parser.add_argument("--fs_parameters",     required=True, nargs="+", type=float, help="JSON list, e.g. [3.0, 0.8]")
    parser.add_argument("--ndensity",          required=True, type=float)
    parser.add_argument("--radbase",           required=True, type=str)
    parser.add_argument("--radparameters",     required=True, nargs="+", type=float, help="JSON list, e.g. [5.5]")
    parser.add_argument("--rcut",              required=True, type=float)
    parser.add_argument("--dcut",              required=True, type=float)
    parser.add_argument("--nrad",              required=True, nargs="+", type=int, help="JSON list, e.g. [8,4,2]")
    parser.add_argument("--lmax",              required=True, nargs="+", type=int, help="JSON list, e.g. [8,6,2]")
    parser.add_argument("--batch_size",        required=True, type=int, help="basis construction batch size for multi-processing")
    parser.add_argument("--out",               required=True, help="Output JSONL path")
    parser.add_argument("--device",            required=True, nargs="+", help="One or more CUDA devices, e.g. cuda:0 cuda:1 cuda:2 cuda:3")
    parser.add_argument("--min_free_gb",       required=True, type=float, help="Minimum GiB of GPU memory needed to allocate device")
    parser.add_argument("--data_path",         required=True, help="f-string path to datasets")
    parser.add_argument("--train_set",         required=True, help="Training set for bandwidth tuning")
    parser.add_argument("--test_sets",         required=True, nargs="+", help="List of test sets")
    parser.add_argument("--labels_path",       required=True, help="Path to JSON file with dataset entropy labels")

    return parser.parse_args()


def parse_cuda_device_index(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":")[1])
    raise ValueError(f"Unsupported device format: {device}")


def query_free_mem_gb_nvidia_smi(dev_idx: int) -> float:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(dev_idx),
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        free_mb = float(out.strip().splitlines()[0])
        return free_mb / 1024.0
    except Exception as e:
        print(f"[cuda:{dev_idx}] Failed to query free mem via nvidia-smi: {e}")
        # Be conservative if we can't query
        return 0.0


def query_total_mem_gb_nvidia_smi(dev_idx: int) -> float:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(dev_idx),
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        total_mb = float(out.strip().splitlines()[0])
        return total_mb / 1024.0
    except Exception as e:
        print(f"[cuda:{dev_idx}] Failed to query total mem via nvidia-smi: {e}")
        return 0.0


def wait_for_gpu_memory(device: str, min_free_gb: float = 45.0, poll_s: float = 60.0):
    dev_idx = parse_cuda_device_index(device)

    total_gb = query_total_mem_gb_nvidia_smi(dev_idx)
    if total_gb <= 0.0:
        print(f"[{device}] Could not determine total memory; skipping wait_for_gpu_memory.")
        return

    # Avoid asking for more than ~80% of the card; prevents impossible targets
    effective_min_free = min(min_free_gb, 0.8 * total_gb)

    while True:
        free_gb = query_free_mem_gb_nvidia_smi(dev_idx)

        if free_gb >= effective_min_free:
            print(
                f"[{device}] Enough free memory: {free_gb:.2f} / {total_gb:.2f} GiB "
                f"(target ≥ {effective_min_free:.2f} GiB)"
            )
            return

        print(
            f"[{device}] Waiting for GPU memory: "
            f"{free_gb:.2f} / {total_gb:.2f} GiB free, "
            f"need ≥ {effective_min_free:.2f} GiB (requested {min_free_gb:.2f} GiB)"
        )
        time.sleep(poll_s)


# TODO: Currently locks a GPU to one process regardless of whether there is enough memory for multiple processes.
@contextmanager
def exclusive_gpu(devices, min_free_gb: float = 45.0, poll_s: float = 60.0):
    if isinstance(devices, str):
        device_list = [devices]
    else:
        device_list = list(devices)

    fd = None
    selected_device = None

    try:
        while True:
            for dev in device_list:
                if not dev.startswith("cuda:"):
                    raise ValueError(f"Unsupported device format: {dev}")
                dev_idx = parse_cuda_device_index(dev)
                lock_path = f"/tmp/entropy_gpu_{dev_idx}.lock"

                fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    print(f"[{dev}] Acquired GPU lock.")
                except BlockingIOError:
                    os.close(fd)
                    fd = None
                    continue

                selected_device = dev
                break

            if selected_device is None:
                print("[GPU] All candidate GPU locks busy, sleeping...")
                time.sleep(poll_s)
                continue

            wait_for_gpu_memory(selected_device, min_free_gb=min_free_gb, poll_s=poll_s)
            yield selected_device
            return

    finally:
        if fd is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            print(f"[{selected_device}] Released GPU lock.")


def make_basis_config(args):
    basis_config = {
        "deltaSplineBins": float(args.deltaSplineBins),
        "elements": args.elements,

        "embeddings": {
            "ALL": {
                "npot": args.npot,
                "fs_parameters": args.fs_parameters,
                "ndensity": int(args.ndensity),
            },
        },

        "bonds": {
            "ALL": {
                "radbase": args.radbase,
                "radparameters": args.radparameters,
                "rcut": args.rcut,
                "dcut": args.dcut,
            }
        },

        "functions": {
            "ALL": {
                "nradmax_by_orders": args.nrad,
                "lmax_by_orders": args.lmax,
            }
        }
    }
    
    return basis_config


def compute_projections(basis, frames):
    desc_dict = compute_B_projections(basis, frames)
    descriptors = desc_dict[0].astype(np.float32)
    return descriptors


def pilot_bandwidth(X, rng=np.random.default_rng(0), max_pts=2000):
    """
    Pilot h0 from the median pairwise *raw* Euclidean distance.
    Works even when feature scales differ (distance will be dominated by large-scale dims).
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    n = len(X)
    idx = np.arange(n)
    if n > max_pts:
        idx = rng.choice(n, size=max_pts, replace=False)
    Y = X[idx]

    m = len(Y)
    # sample random pairs (no full O(n^2) matrix)
    k = min(20000, m * (m - 1) // 2)
    i1 = rng.integers(0, m, size=k)
    i2 = rng.integers(0, m, size=k)
    mask = i1 != i2
    i1, i2 = i1[mask], i2[mask]

    dists = np.linalg.norm(Y[i1] - Y[i2], axis=1)
    med = np.median(dists)
    d = X.shape[1]
    # Same mapping as before, but now in RAW space
    h0 = med / np.sqrt(2.0 * max(d, 1))
    if not np.isfinite(h0) or h0 <= 0:
        n_eff = len(X)
        h0 = n_eff ** (-1.0 / (max(d, 1) + 4))
    print(f"pilot bandwidth {h0}")
    return float(h0)


def evaluate_entropy_loss(X, S_star, h, batch_size=10000, device="cpu"):
    S = entropy(X, h=h, batch_size=batch_size, device=device)
    print(f"entropy {S} bandwidth {h}")
    return (S - S_star) ** 2, S


def coarse_log_grid_bracket(X, S_star, h0, width_factor=100.0, num=25, batch_size=10000, device="cpu"):
    print("Starting scan...")
    lo = np.log10(h0 / width_factor)
    hi = np.log10(h0 * width_factor)
    grid = np.linspace(lo, hi, num)
    vals = []
    for t in grid:
        h = 10.0 ** t
        f, Sval = evaluate_entropy_loss(X, S_star, h, batch_size=batch_size, device=device)
        vals.append((t, f, Sval))
    best_i = int(np.argmin([v[1] for v in vals]))
    a_i = max(0, best_i - 1)
    c_i = min(len(vals) - 1, best_i + 1)
    if a_i == best_i:
        a_i = max(0, best_i - 2)
    if c_i == best_i:
        c_i = min(len(vals) - 1, best_i + 2)
    a, fa, _ = vals[a_i]
    b, fb, _ = vals[best_i]
    c, fc, _ = vals[c_i]
    if not (fb <= fa and fb <= fc):
        a, fa, _ = vals[max(0, best_i - 1)]
        c, fc, _ = vals[min(len(vals) - 1, best_i + 1)]
    return (a, fa), (b, fb), (c, fc), vals


def golden_section_search_log10(X, S_star, a, b, c, max_iter=60, tol=1e-3, batch_size=10000, device="cpu"):
    print("Starting search...")
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    left, right = a, c
    x1 = right - gr * (right - left)
    x2 = left + gr * (right - left)

    def f_of_t(t):
        h = 10.0 ** t
        return evaluate_entropy_loss(X, S_star, h, batch_size=batch_size, device=device)

    f1, S1 = f_of_t(x1)
    f2, S2 = f_of_t(x2)
    for _ in range(max_iter):
        if abs(right - left) < tol:
            break
        if f1 > f2:
            left = x1
            x1, f1, S1 = x2, f2, S2
            x2 = left + gr * (right - left)
            f2, S2 = f_of_t(x2)
        else:
            right = x2
            x2, f2, S2 = x1, f1, S1
            x1 = right - gr * (right - left)
            f1, S1 = f_of_t(x1)

    if f1 < f2:
        return x1, 10.0 ** x1, f1, S1
    else:
        return x2, 10.0 ** x2, f2, S2


def optimize_bandwidth_entropy(X, S_star, batch_size=10000, grid_width=100.0, grid_pts=25, device="cpu"):
    h0 = pilot_bandwidth(X)
    (a, fa), (b, fb), (c, fc), scan = coarse_log_grid_bracket(
        X, S_star, h0, width_factor=grid_width, num=grid_pts, batch_size=batch_size, device=device,
    )
    t_best, h_best, f_best, S_best = golden_section_search_log10(
        X, S_star, a, b, c, max_iter=60, tol=1e-3, batch_size=batch_size, device=device,
    )
    report = {
        "h0": h0,
        "log10_bounds": (a, c),
        "grid_points": grid_pts,
        "best_log10h": t_best,
        "best_h": h_best,
        "best_entropy": S_best,
        "target_entropy": S_star,
        "abs_error": abs(S_best - S_star),
        "squared_error": f_best,
    }
    return h_best, report


def main():
    args = parse_args()
    print(args)
    basis_config = make_basis_config(args)

    out = args.out
    device = args.device
    min_free_gb = args.min_free_gb
    data_path = args.data_path
    train_set = args.train_set
    test_sets = args.test_sets
    print(train_set)
    print(test_sets)
    batch_size = args.batch_size
    with open(args.labels_path, "r") as f:
        labels = json.load(f)
    print(f"nrad {args.nrad} lmax {args.lmax}")

    basis = create_multispecies_basis_config(basis_config)
    train_frames_list = read(data_path.format(data_name=train_set), index=":")

    n_batches = int(np.ceil(len(train_frames_list) / batch_size))
    batches = [
        train_frames_list[i:i + batch_size]
        for i in range(0, len(train_frames_list), batch_size)
    ]
    # Uses multi-processing
    X_train = Parallel(n_jobs=n_batches)(
        delayed(compute_projections)(basis, batch) for batch in batches
    )
    X_train = np.concatenate(X_train)
    print(f"X_train shape {X_train.shape}")

    with exclusive_gpu(device, min_free_gb=min_free_gb, poll_s=30.0) as gpu:
        X_train_tensor = torch.tensor(X_train, device=gpu)

        h_opt, opt_report = optimize_bandwidth_entropy(
            X_train_tensor,
            S_star=labels[train_set],
            batch_size=10000,
            grid_width=100.0,
            grid_pts=25,
            device=gpu,
        )

        print(f"\n=== Bandwidth optimization ({train_set}) ===")
        print(f"pilot h0         : {opt_report['h0']:.6g}")
        print(f"search log10 span: [{opt_report['log10_bounds'][0]:.3f}, {opt_report['log10_bounds'][1]:.3f}]")
        print(f"best h           : {opt_report['best_h']:.6g}")
        print(f"S(best h)        : {opt_report['best_entropy']:.9f}")
        print(f"target S*        : {opt_report['target_entropy']:.9f}")
        print(f"|S - S*|         : {opt_report['abs_error']:.6g}")

        # Free GPU memory from training
        del X_train_tensor, train_frames_list
        torch.cuda.empty_cache()

    entry = {
        "basis_config": basis_config,
        "bandwidth": h_opt,
        "entropies": {}
    }

    X_tests_dict = {}
    for test_set in test_sets:
        print(test_set)
        if train_set == test_set:
            X_test = X_train
        else:
            test_frames_list = read(data_path.format(data_name=test_set), index=":")
        
            n_batches = int(np.ceil(len(test_frames_list) / batch_size))
            batches = [
                test_frames_list[i:i + batch_size]
                for i in range(0, len(test_frames_list), batch_size)
            ]
            # Uses multi-processing
            X_test = Parallel(n_jobs=n_batches)(
                delayed(compute_projections)(basis, batch) for batch in batches
            )
            X_test = np.concatenate(X_test)
            del test_frames_list
        print(f"X_test shape {X_test.shape}")
        X_tests_dict[test_set] = X_test
        

    with exclusive_gpu(device, min_free_gb=min_free_gb, poll_s=30.0) as gpu:
        for test_set in test_sets:
            X_test = X_tests_dict[test_set]
            X_test_tensor = torch.tensor(X_test, device=gpu)
            print("entropy")
            S = entropy(X_test_tensor, h=h_opt, batch_size=10000, device=gpu)
            print(S)
            entry["entropies"][test_set] = S.item()
        # Free test set from GPU memory
        del X_test_tensor
        torch.cuda.empty_cache()

    with open(out, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)
    
    print("Run completed.")

if __name__ == "__main__":
    main()