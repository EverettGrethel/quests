import numpy as np
import torch
from quests.gpu.entropy import entropy, entropy_cosine


def evaluate_entropy_loss(X, S_star, h, batch_size=10000, cosine=False, device="cpu"):
    if not cosine:
        S = entropy(X, h=h, batch_size=batch_size, device=device)
    else:
        S = entropy_cosine(X, h=h, batch_size=batch_size, device=device)
    return (S - S_star) ** 2, S


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

def coarse_log_grid_bracket(
    X, S_star, h0,
    width_factor=100.0,
    num=50,
    max_expand=10,
    batch_size=10000,
    cosine=False,
    device="cpu",
):
    # Expand until the minimum is not on the boundary
    for _ in range(max_expand):
        lo = np.log10(h0 / width_factor)
        hi = np.log10(h0 * width_factor)
        grid = np.linspace(lo, hi, num)

        vals = []
        for t in grid:
            h = 10.0 ** t
            f, Sval = evaluate_entropy_loss(X, S_star, h, batch_size=batch_size, cosine=cosine, device=device)
            vals.append((t, float(f), float(Sval)))

        best_i = int(np.argmin([v[1] for v in vals]))

        # If best is on an edge, expand and try again
        if best_i == 0 or best_i == len(vals) - 1:
            width_factor *= 10.0
            continue

        # Return a proper bracket around the best interior point
        a, fa, _ = vals[best_i - 1]
        b, fb, _ = vals[best_i]
        c, fc, _ = vals[best_i + 1]
        return (a, fa), (b, fb), (c, fc), vals, width_factor

    raise RuntimeError("Could not find an interior minimum; expanded scan too many times.")



def golden_section_search_log10(X, S_star, a, b, c, max_iter=60, tol=1e-3, batch_size=10000, cosine=False, device="cpu"):
    print("Starting search...")
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    left, right = a, c
    x1 = right - gr * (right - left)
    x2 = left + gr * (right - left)

    def f_of_t(t):
        h = 10.0 ** t
        return evaluate_entropy_loss(X, S_star, h, batch_size=batch_size, cosine=cosine, device=device)

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


def optimize_bandwidth_entropy(
        X,
        S_star,
        batch_size=10000,
        grid_width=100.0,
        grid_pts=25,
        cosine=False,
        device="cpu"
        ):
    h0 = pilot_bandwidth(X)
    (a, fa), (b, fb), (c, fc), scan, width_factor = coarse_log_grid_bracket(
        X, S_star, h0, width_factor=grid_width, num=grid_pts, batch_size=batch_size, cosine=cosine, device=device,
    )
    t_best, h_best, f_best, S_best = golden_section_search_log10(
        X, S_star, a, b, c, max_iter=60, tol=1e-3, batch_size=batch_size, cosine=cosine, device=device,
    )
    report = {
        "h0": h0,
        "log10_bounds": (a, c),
        "grid_points": grid_pts,
        "grid_width": width_factor,
        "best_log10h": t_best,
        "best_h": h_best,
        "best_entropy": S_best,
        "target_entropy": S_star,
        "abs_error": abs(S_best - S_star),
        "squared_error": f_best,
    }
    return h_best, report
