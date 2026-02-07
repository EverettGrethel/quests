import json
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
from collections import defaultdict
from pathlib import Path

# -------------------------------
# Paths
# -------------------------------
sweep_path = Path(f"/home/grethel/dev/quests/sweep_results/sweep_ace_Graphite_cosine.jsonl")
reference_path = Path("/home/grethel/dev/quests/gap20_quests_entropy.json")
out_dir = Path("sweep_plots")
out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Load sweep data
# -------------------------------
with open(sweep_path, "r") as f:
    entries = [json.loads(line) for line in f]
entries = sorted(entries, key=lambda e: e["features"], reverse=True)

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

# -------------------------------
# Color mapping by k
# -------------------------------
feature_vals = [e["features"] for e in entries]
norm = mcolors.LogNorm(
    vmin=min(feature_vals),
    vmax=max(feature_vals)
)

cmap = cm.viridis

def curve_label(e):
    return f"features={e['features']} nradmax={e['basis_config']['functions']['ALL']['nradmax_by_orders']} lmax={e['basis_config']['functions']['ALL']['lmax_by_orders']}"

# -------------------------------
# 1) Plot all curves
# -------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for e in entries:
    y = [e["entropies"][k] for k in categories]
    color = cmap(norm(e["features"]))
    ax.plot(
        categories,
        y,
        marker="o",
        color=color,
        label=curve_label(e),
        alpha=0.8
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
ax.set_title("ACE Entropy Sweep (cosine distance)")
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, rotation=30, ha="right")

ax.legend(title="Config", bbox_to_anchor=(1.15, 1), loc="upper left")

# Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("k")

fig.tight_layout()
fig.savefig(out_dir / f"ace_Graphite_entropy_cosine.png", dpi=200, bbox_inches="tight")
plt.close(fig)
