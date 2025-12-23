import json
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from pathlib import Path
from collections import defaultdict

# --------------------------------------------------
# Paths
# --------------------------------------------------
data_path = Path("/home/grethel/dev/quests/sweep_results/sweep_quests_Graphite_strain.jsonl")
reference_path = Path("/home/grethel/dev/quests/gap20_quests_entropy.json")
out_path = Path("sweep_plots/quests_entropy_strain_rainbow.png")

# --------------------------------------------------
# Load data
# --------------------------------------------------
with open(data_path) as f:
    entries = [json.loads(line) for line in f]

with open(reference_path) as f:
    reference = json.load(f)

# --------------------------------------------------
# Determine category order
# --------------------------------------------------
categories = list(entries[0]["entropies"].keys())
reference_y = [reference.get(k, None) for k in categories]

# --------------------------------------------------
# Color mapping by strain (larger strain → more red)
# --------------------------------------------------
strain_vals = [e["strain"] for e in entries]
norm = mcolors.Normalize(vmin=min(strain_vals), vmax=max(strain_vals))
cmap = cm.rainbow

# --------------------------------------------------
# Marker size mapping by k
# --------------------------------------------------
k_vals = [e["k"] for e in entries]
k_min, k_max = min(k_vals), max(k_vals)

def marker_size(k, smin=40, smax=160):
    """Scale marker radius with k"""
    return smin + (k - k_min) / (k_max - k_min) * (smax - smin)

# --------------------------------------------------
# Label helper
# --------------------------------------------------
def curve_label(e):
    return f"strain={e['strain']} | k={e['k']} | cutoff={e['cutoff']} | features={e['features']}"

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for e in entries:
    y = [e["entropies"][k] for k in categories]
    color = cmap(norm(e["strain"]))
    size = marker_size(e["k"])

    ax.plot(
        categories,
        y,
        marker="o",
        markersize=size / 10,  # matplotlib uses diameter-ish scaling
        linewidth=1.5,
        color=color,
        label=curve_label(e),
        alpha=0.85,
    )

# --------------------------------------------------
# Reference QUESTS curve (black)
# --------------------------------------------------
ax.plot(
    categories,
    reference_y,
    color="black",
    linewidth=3.0,
    marker="o",
    markersize=8,
    label="QUESTS (reference)",
    zorder=10,
)

# --------------------------------------------------
# Axes formatting
# --------------------------------------------------
ax.set_xlabel("Dataset")
ax.set_ylabel("Entropy")
ax.set_title("Dataset Entropy (QUESTS)")

ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, rotation=30, ha="right")

# --------------------------------------------------
# Legend & colorbar
# --------------------------------------------------
ax.legend(
    title="Config",
    bbox_to_anchor=(1.18, 1.0),
    loc="upper left",
    fontsize=9,
)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("Strain")

# --------------------------------------------------
# Save
# --------------------------------------------------
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------
# Grouped plots: same k, different strain
# --------------------------------------------------

grouped_by_k = defaultdict(list)
for e in entries:
    grouped_by_k[e["k"]].append(e)

for k_val, k_entries in sorted(grouped_by_k.items()):
    fig, ax = plt.subplots(figsize=(10, 6))

    for e in sorted(k_entries, key=lambda x: x["strain"]):
        y = [e["entropies"][c] for c in categories]
        color = cmap(norm(e["strain"]))

        ax.plot(
            categories,
            y,
            marker="o",
            markersize=marker_size(k_val) / 10,
            linewidth=1.8,
            color=color,
            alpha=0.9,
            label=f"strain={e['strain']}",
        )

    # Reference QUESTS curve
    ax.plot(
        categories,
        reference_y,
        color="black",
        linewidth=3.0,
        marker="o",
        markersize=8,
        label="QUESTS (reference)",
        zorder=10,
    )

    # Formatting
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Entropy")
    ax.set_title(f"Dataset Entropy (QUESTS) — k={k_val}")

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=30, ha="right")

    ax.legend(
        title="Strain",
        bbox_to_anchor=(1.18, 1.0),
        loc="upper left",
        fontsize=9,
    )

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Strain")

    # Save
    fig.tight_layout()
    out_k = out_path.with_name(f"quests_entropy_k_{k_val}_strain_rainbow.png")
    fig.savefig(out_k, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved grouped plot → {out_k}")


print(f"Saved plot → {out_path}")
