import json
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
from pathlib import Path

# ----------------------------
# QUESTS reference values
# ----------------------------
reference = {
    "Graphene": 4.245179458166078,
    "Diamond": 4.318381910272738,
    "Graphite": 5.6085074467370095,
    "Nanotubes": 7.0282707526691715,
    "Fullerenes": 8.67911004440742,
    "Liquid": 11.61485589283075
}

# ----------------------------
# Load JSONL data
# ----------------------------
data_path = "entropy_results.jsonl"

with open(data_path, "r") as f:
    entries = [json.loads(line) for line in f]

# Dataset categories from the new data
categories = list(entries[0]["entropy"].keys())

# Restrict reference to categories actually present
reference_y = [reference[k] for k in categories]

# ----------------------------
# Colormap: features → rainbow
# ----------------------------
features_vals = [e["features"] for e in entries]
norm = mcolors.Normalize(vmin=min(features_vals), vmax=max(features_vals))
cmap = cm.rainbow

# ----------------------------
# Legend label
# ----------------------------
def label_entry(e):
    return (f"rcut={e['rcut']}, "
            f"order={e['order']}, "
            f"degree={e['totaldegree']}, "
            f"features={e['features']}")

# ----------------------------
# Create figure (all curves)
# ----------------------------
fig, ax = plt.subplots(figsize=(11, 7))

# Plot all configuration curves
for e in entries:
    y = [e["entropy"][k] for k in categories]
    color = cmap(norm(e["features"]))
    ax.plot(categories, y, marker="o", color=color, label=label_entry(e))

# ----------------------------
# Add QUESTS reference curve (black)
# ----------------------------
ax.plot(categories, reference_y,
        color="black", linewidth=3, marker="o",
        label="QUESTS", zorder=10)

# ----------------------------
# Axes formatting
# ----------------------------
ax.set_title("Dataset Entropy – All Configurations", fontsize=14)
ax.set_xlabel("Dataset", fontsize=12)
ax.set_ylabel("Entropy", fontsize=12)
ax.set_xticklabels(categories, rotation=30, ha="right")

# ----------------------------
# Legend
# ----------------------------
handles, labels = ax.get_legend_handles_labels()

# Put QUESTS at the top of legend
sorted_handles = [handles[-1]] + handles[:-1]
sorted_labels = ["QUESTS"] + labels[:-1]

ax.legend(
    sorted_handles,
    sorted_labels,
    title="Configurations",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=9
)

# ----------------------------
# Colorbar for features
# ----------------------------
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("features")

fig.tight_layout()

# Save image
Path("sweep_plots").mkdir(exist_ok=True)
output_file = Path("sweep_plots/entropy_all_curves_with_quests_rainbow.png")
fig.savefig(output_file, dpi=200, bbox_inches="tight")

print(f"Saved: {output_file}")
