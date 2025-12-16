import json
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
from collections import defaultdict
from pathlib import Path

# Reference values (same as your script; safe if some keys missing)
reference = {
    "Graphene": 4.245179458166078,
    "Diamond": 4.318381910272738,
    "Graphite": 5.6085074467370095,
    "Nanotubes": 7.0282707526691715,
    "Fullerenes": 8.67911004440742,
    "Defects": 9.531933892473084,
    "Surfaces": 9.823139796211981,
    "Liquid": 11.61485589283075,
    "Amorphous_Bulk": 12.183809856122803,
    "methane_subset": 9.11230634401734,
}

data = "graphite"
# data = "methane_subset"
data_path = f"/home/grethel/dev/quests/sweep_results/sweep_soap_{data}.jsonl"

with open(data_path, 'r') as file:
    entries = [json.loads(line) for line in file]

# Extract category order from the first entry
categories = list(entries[0]["entropies"].keys())
reference_y = [reference.get(k, None) for k in categories]

# Map color by "features"
feature_vals = [e["features"] for e in entries]
norm = mcolors.Normalize(vmin=min(feature_vals), vmax=max(feature_vals))
cmap = cm.rainbow

def curve_label(e):
    return (f"feat={e['features']} | r_cut={e['r_cut']} | "
            f"n_max={e['n_max']} | l_max={e['l_max']}")

# -------------------------------
# 1) Plot all curves
# -------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for e in entries:
    y = [e["entropies"][k] for k in categories]
    color = cmap(norm(e["features"]))
    ax.plot(categories, y, marker="o", color=color, label=curve_label(e))

# Reference curve
ax.plot(categories, reference_y, color="black", linewidth=2.5,
        marker="o", label="QUESTS", zorder=10)

ax.set_xlabel("Dataset")
ax.set_ylabel("Entropy")
ax.set_title(f"Dataset Entropy (SOAP)")
ax.set_xticklabels(categories, rotation=30, ha="right")
ax.legend(title="Config", bbox_to_anchor=(1.15, 1), loc="upper left")

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, pad=0.02)

fig.tight_layout()
out_all = Path(f"sweep_plots/soap_{data}_entropy_all_curves_features_rainbow.png")
fig.savefig(out_all, dpi=200, bbox_inches="tight")
plt.close(fig)

# -------------------------------
# 2) Group by (r_cut, n_max, l_max)
# -------------------------------
grouped = defaultdict(list)
for e in entries:
    key = (e["r_cut"], e["n_max"], e["l_max"])
    grouped[key].append(e)

out_paths = []
for (r_cut, n_max, l_max), lst in sorted(grouped.items()):
    fig, ax = plt.subplots(figsize=(10, 6))

    for e in lst:
        y = [e["entropies"][k] for k in categories]
        color = cmap(norm(e["features"]))
        label = f"feat={e['features']} | n_max={n_max} | l_max={l_max}"
        ax.plot(categories, y, marker="o", color=color, label=label)

    # Reference curve
    ax.plot(categories, reference_y, color="black", linewidth=2.5,
            marker="o", label="QUESTS", zorder=10)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Entropy")
    ax.set_title(f"Entropy curves (r_cut={r_cut}, n_max={n_max}, l_max={l_max})")
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.legend(title="Config", bbox_to_anchor=(1.15, 1), loc="upper left")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
