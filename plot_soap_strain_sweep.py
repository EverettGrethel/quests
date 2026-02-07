import json
import math
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from collections import defaultdict
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
train_set = "Graphite"
data_path = Path(f"sweep_results/sweep_soap_{train_set}_strain.jsonl")
reference_path = Path("/home/grethel/dev/quests/gap20_quests_entropy.json")  # QUESTS reference
out_path = Path(f"sweep_plots/soap_entropy_{train_set}_strain_grid.png")

# -----------------------------
# Load entries + reference
# -----------------------------
with open(data_path) as f:
    entries = [json.loads(line) for line in f]

with open(reference_path) as f:
    reference = json.load(f)

# -----------------------------
# Category order (x-axis)
# -----------------------------
categories = list(entries[0]["entropies"].keys())
reference_y = [reference.get(c, None) for c in categories]

# -----------------------------
# Global y-axis limits
# -----------------------------
all_entropy_vals = []

for e in entries:
    all_entropy_vals.extend(e["entropies"].values())

all_entropy_vals.extend(reference_y)

y_max = max(all_entropy_vals)

y_limits = (0.0, y_max * 1.05)  # 5% headroom at the top


# -----------------------------
# Group by (n_max, l_max)
# -----------------------------
groups = defaultdict(list)
for e in entries:
    groups[(e["n_max"], e["l_max"])].append(e)

group_keys = sorted(groups.keys())  # sorted (n_max, l_max)

# -----------------------------
# Marker map by r_cut
# -----------------------------
r_cuts = sorted({e["r_cut"] for e in entries})
marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
r_cut_to_marker = {rc: marker_cycle[i % len(marker_cycle)] for i, rc in enumerate(r_cuts)}

# -----------------------------
# Color map by strain (higher strain -> more red)
# -----------------------------
strain_vals = [e["strain"] for e in entries]
norm = mcolors.Normalize(vmin=min(strain_vals), vmax=max(strain_vals))
cmap = cm.viridis  # reversed rainbow so max strain looks red

# -----------------------------
# Subplot layout
# -----------------------------
n_panels = len(group_keys)
ncols = min(3, n_panels)
nrows = math.ceil(n_panels / ncols)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(5.5 * ncols, 4.2 * nrows),
    sharey=True
)

if n_panels == 1:
    axes = [axes]
else:
    axes = axes.flatten()

# Main title
fig.suptitle("SOAP Dataset Entropy (Strain)", fontsize=16, y=0.98)

# -----------------------------
# Plot each (n_max, l_max) panel
# -----------------------------
for ax, (nmax, lmax) in zip(axes, group_keys):
    panel_entries = sorted(groups[(nmax, lmax)], key=lambda x: (x["strain"], x["r_cut"]))

    for e in panel_entries:
        y = [e["entropies"][c] for c in categories]
        color = cmap(norm(e["strain"]))
        marker = r_cut_to_marker[e["r_cut"]]

        ax.set_ylim(*y_limits)

        ax.plot(
            categories,
            y,
            color=color,
            marker=marker,
            linewidth=1.6,
            markersize=6,
            alpha=0.9,
        )

    # Reference QUESTS curve (black) — add back to every subplot
    ax.plot(
        categories,
        reference_y,
        color="black",
        linewidth=2.8,
        marker="o",
        markersize=6,
        label="QUESTS (reference)",
        zorder=10,
    )

    ax.set_title(f"SOAP: n_max={nmax}, l_max={lmax}")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.grid(True, alpha=0.25)

# Hide unused axes if grid is bigger than number of panels
for ax in axes[len(group_keys):]:
    ax.set_visible(False)

# Common labels
fig.supylabel("Entropy")
fig.supxlabel("Dataset")

# -----------------------------
# Layout: reserve space on the right for colorbar + legend
# (prevents covering the rightmost subplot)
# -----------------------------
fig.subplots_adjust(right=0.82, top=0.90)  # right margin for cbar/legend; top for suptitle

# -----------------------------
# Strain legend (color-coded)
# -----------------------------
strain_levels = sorted({e["strain"] for e in entries})

strain_handles = [
    plt.Line2D(
        [0], [0],
        color=cmap(norm(s)),
        marker="o",
        linestyle="-",
        linewidth=2.0,
        markersize=6,
        label=f"strain={s}",
    )
    for s in strain_levels
]

# -----------------------------
# Add marker legend for r_cut (figure-level)
# -----------------------------
# -----------------------------
# Marker legend (r_cut) + reference
# -----------------------------
marker_handles = [
    plt.Line2D(
        [0], [0],
        color="black",
        marker=r_cut_to_marker[rc],
        linestyle="None",
        markersize=7,
        label=f"r_cut={rc}",
    )
    for rc in r_cuts
]

ref_handle = plt.Line2D(
    [0], [0],
    color="black",
    linewidth=2.8,
    marker="o",
    markersize=6,
    label="QUESTS (reference)",
)


# Strain legend (top-right)
fig.legend(
    handles=strain_handles,
    title="Color = Strain",
    loc="upper left",
    bbox_to_anchor=(0.84, 0.98),
    fontsize=9,
)

# Marker legend (below strain legend)
fig.legend(
    handles=[ref_handle] + marker_handles,
    title="Marker = r_cut",
    loc="upper left",
    bbox_to_anchor=(0.84, 0.58),
    fontsize=9,
)


# Use tight_layout but keep our manual right margin/colorbar axes
fig.tight_layout(rect=[0.0, 0.0, 0.82, 0.92])

fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved → {out_path}")
