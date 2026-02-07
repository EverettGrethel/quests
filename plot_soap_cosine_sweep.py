import json
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
import numpy as np
from matplotlib.colors import BoundaryNorm

from collections import defaultdict
from pathlib import Path

# -------------------------------
# Paths
# -------------------------------
train_set = "Graphite"
sweep_path = Path(f"/home/grethel/dev/quests/sweep_results/sweep_soap_{train_set}_cosine.jsonl")
reference_path = Path("/home/grethel/dev/quests/gap20_quests_entropy.json")
out_dir = Path("sweep_plots")
out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Load sweep data
# -------------------------------
with open(sweep_path, "r") as f:
    entries = [json.loads(line) for line in f]
    entries = sorted(entries, key=lambda e: e["features"])

# -------------------------------
# Load reference data
# -------------------------------
with open(reference_path, "r") as f:
    reference_entropies = json.load(f)

# -------------------------------
# Category order
# -------------------------------
categories = list(entries[0]["entropies"].keys())
reference_y = [reference_entropies.get(k, None) for k in categories]

# Discrete, high-contrast mapping (rank-based)
unique_features = sorted(set(e["features"] for e in entries))
n = len(unique_features)

cmap = cm.get_cmap("viridis", n)  # or "tab20" if n <= 20
feature_to_color = {f: cmap(i) for i, f in enumerate(unique_features)}

def curve_label(e):
    return (
        f"features={e['features']} | "
        f"r_cut={e['r_cut']} | "
        f"n_max={e['n_max']} | "
        f"l_max={e['l_max']}"
    )

linestyles = ["-", "--", ":", "-."]
rcut_markers = {
    5.0: "o",   # circle
    6.0: "s",   # square
    7.0: "D",   # diamond
}

# -------------------------------
# 1) Plot all curves
# -------------------------------

from matplotlib.lines import Line2D

legend_handles = []

# r_cut legend entries (marker-shape based)
for rc in sorted(set(e["r_cut"] for e in entries)):
    marker = rcut_markers.get(rc, "o")
    legend_handles.append(
        Line2D(
            [0], [0],
            marker=marker,
            linestyle="None",
            color="gray",
            label=f"r_cut={rc}",
            markersize=7,
        )
    )


fig, ax = plt.subplots(figsize=(10, 6))

for e in entries:
    y = [e["entropies"][k] for k in categories]
    color = feature_to_color[e["features"]]
    ls = linestyles[int(round(e["r_cut"])) % len(linestyles)]

    ax.plot(
        categories,
        y,
        marker="o",
        color=color,
        label=curve_label(e),
        alpha=0.85,
        linestyle=ls,
    )

# Reference curve
ax.plot(
    categories,
    reference_y,
    color="black",
    linewidth=2.8,
    marker="o",
    label="QUESTS",
    zorder=10
)

ax.set_xlabel("Dataset")
ax.set_ylabel("Entropy")
ax.set_title("SOAP Entropy Sweep (cosine distance)")
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, rotation=30, ha="right")

ax.legend(title="SOAP config", bbox_to_anchor=(1.2, 1), loc="upper left")

# Colorbar
bounds = np.arange(n + 1) - 0.5
norm = BoundaryNorm(bounds, n)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, pad=0.02, ticks=np.arange(n))
cbar.ax.set_yticklabels(unique_features)
cbar.set_label("Number of SOAP features")

fig.tight_layout()
fig.savefig(out_dir / "soap_entropy_Graphite_cosine.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# # -------------------------------
# # 2) Group by dataset
# # -------------------------------
# -------------------------------
# 2) One-column "zoomed" plots: one figure per dataset column
# -------------------------------

# Helpful y-limits so every single-column plot shares the same scale


for dataset in categories:
    fig, ax = plt.subplots(figsize=(6, 8))

    # Dataset-specific y-limits
    ys = [e["entropies"][dataset] for e in entries if e["entropies"].get(dataset) is not None]

    # ref_y = reference_entropies.get(dataset, None)
    # if ref_y is not None:
    #     ys.append(ref_y)

    ymin, ymax = min(ys), max(ys)
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)

    # Plot one point per entry at the single x position
    for e in entries:
        y = e["entropies"][dataset]
        color = feature_to_color[e["features"]]
        # ls = linestyles[int(round(e["r_cut"])) % len(linestyles)]
        marker = rcut_markers.get(e["r_cut"], "o")

        # Use linestyle in marker edge/size to keep some encoding but avoid "lines"
        ax.plot(
            [dataset],
            [y],
            marker=marker,
            linestyle="None",   # IMPORTANT: no curve, just stacked points
            color=color,
            alpha=0.9,
            markersize=6,
        )

    # # Reference point for this dataset (if available)
    # ref_y = reference_entropies.get(dataset, None)
    # if ref_y is not None:
    #     ax.plot(
    #         [dataset],
    #         [ref_y],
    #         color="black",
    #         marker="o",
    #         markersize=8,
    #         linestyle="None",
    #         label="Reference",
    #         zorder=10
    #     )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Entropy")
    ax.set_title(f"QUESTS Entropy Sweep (SOAP) — {dataset}")

    # keep the single x tick, rotated like before
    ax.set_xticks([0])
    ax.set_xticklabels([dataset], rotation=30, ha="right")


    # Colorbar
    bounds = np.arange(n + 1) - 0.5
    norm = BoundaryNorm(bounds, n)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.02, ticks=np.arange(n))
    cbar.ax.set_yticklabels(unique_features)
    cbar.set_label("Number of SOAP features")

    ax.legend(
        handles=legend_handles,
        title="Config",
        loc="upper right",
        frameon=True
    )

    fig.tight_layout()
    title = out_dir / f"soap_entropy_{train_set}_cosine_test_{dataset}.png"
    print(title)
    fig.savefig(title, dpi=200, bbox_inches="tight")
    plt.close(fig)
