import argparse
import json
import numpy as np
import torch

from ase.io import read
from pyace import create_multispecies_basis_config
from pyace.activelearning import compute_B_projections
from quests.gpu.entropy import entropy


def print_cuda_memory():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    # Current GPU
    dev = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(dev) / (1024**3)
    reserved  = torch.cuda.memory_reserved(dev)  / (1024**3)
    peak      = torch.cuda.max_memory_allocated(dev) / (1024**3)

    print(f"\n=== CUDA Memory ===")
    print(f"Allocated: {allocated:.2f} GiB")
    print(f"Reserved : {reserved:.2f} GiB")
    print(f"Peak     : {peak:.2f} GiB\n")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--elements",          required=True, help='JSON list, e.g. ["C"]')
    parser.add_argument("--deltaSplineBins",   required=True, type=str)
    parser.add_argument("--npot",              required=True, type=str)
    parser.add_argument("--fs_parameters",     required=True, help="JSON list, e.g. [3.0, 0.8]")
    parser.add_argument("--ndensity",          required=True, type=float)
    parser.add_argument("--radbase",           required=True, type=str)
    parser.add_argument("--radparameters",     required=True, help="JSON list, e.g. [5.5]")
    parser.add_argument("--rcut",              required=True, type=float)
    parser.add_argument("--dcut",              required=True, type=float)
    parser.add_argument("--nrad",              required=True, help="JSON list, e.g. [8,4,2]")
    parser.add_argument("--lmax",              required=True, help="JSON list, e.g. [8,6,2]")
    parser.add_argument("--out",               required=True, help="Output JSONL path")
    parser.add_argument("--device",            required=True, help="Device name")
    parser.add_argument("--data_path",         required=True, help="f-string path to datasets")
    parser.add_argument("--train_set",         required=True, help="Training set for bandwidth tuning")
    parser.add_argument("--test_sets",         required=True, help="List of test sets")
    parser.add_argument("--labels",            required=True, help="Dictionary of dataset entropy labels")

    return parser.parse_args()


def make_basis_config(args):
    elements = json.loads(args.elements)
    fs_parameters = json.loads(args.fs_parameters)
    radparameters = json.loads(args.radparameters)
    nrad = json.loads(args.nrad)
    lmax = json.loads(args.lmax)

    basis_config = {
        "deltaSplineBins": float(args.deltaSplineBins),
        "elements": elements,

        "embeddings": {
            "ALL": {
                "npot": args.npot,
                "fs_parameters": fs_parameters,
                "ndensity": int(args.ndensity),
            },
        },

        "bonds": {
            "ALL": {
                "radbase": args.radbase,
                "radparameters": radparameters,
                "rcut": args.rcut,
                "dcut": args.dcut,
            }
        },

        "functions": {
            "ALL": {
                "nradmax_by_orders": nrad,
                "lmax_by_orders": lmax,
            }
        }
    }
    
    return basis_config


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
    basis_config = make_basis_config(args)

    out = args.out
    device = args.device
    data_path = args.data_path
    train_set = args.train_set
    test_sets = json.loads(args.test_sets)
    labels = json.loads(args.labels)
    print(f"nrad {json.loads(args.nrad)} lmax {json.loads(args.lmax)}")

    basis = create_multispecies_basis_config(basis_config)

    train_frames_list = read(data_path.format(data_name=train_set), index=":")
    X_train = compute_B_projections(basis, train_frames_list)[0].astype(np.float32)
    X_train = torch.tensor(X_train, device=device)

    h_opt, opt_report = optimize_bandwidth_entropy(
        X_train, S_star=labels[train_set], batch_size=10000, grid_width=100.0, grid_pts=25, device=device
    )

    print("\n=== Bandwidth optimization (Graphite) ===")
    print(f"pilot h0         : {opt_report['h0']:.6g}")
    print(f"search log10 span: [{opt_report['log10_bounds'][0]:.3f}, {opt_report['log10_bounds'][1]:.3f}]")
    print(f"best h           : {opt_report['best_h']:.6g}")
    print(f"S(best h)        : {opt_report['best_entropy']:.9f}")
    print(f"target S*        : {opt_report['target_entropy']:.9f}")
    print(f"|S - S*|         : {opt_report['abs_error']:.6g}")

    # Free GPU memory from training
    del X_train, train_frames_list
    if device != "cpu" and "cuda" in device:
        torch.cuda.empty_cache()

    entry = {
        "basis_config": basis_config,
        "bandwidth": h_opt,
        "entropies": {}
    }

    for test_set in test_sets:
        print(test_set)
        test_frames_list = read(data_path.format(data_name=test_set), index=":")
        print("projections")
        X_test = compute_B_projections(basis, test_frames_list)[0].astype(np.float32)
        X_test = torch.tensor(X_test, device=device)
        print("entropy")
        S = entropy(X_test, h=h_opt, batch_size=10000, device=device)
        print(S)
        entry["entropies"][test_set] = S.item()

        # Free GPU memory from test
        del X_test, test_frames_list
        if device != "cpu" and "cuda" in device:
            torch.cuda.empty_cache()

    with open(out, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()

if __name__ == "__main__":
    main()