import json
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from pathlib import Path
from collections import defaultdict

# --------------------------------------------------
# Paths
# --------------------------------------------------
data_path = Path("/home/grethel/dev/quests/sweep_results/sweep_soap_Graphite_strain.jsonl")
reference_path = Path("/home/grethel/dev/quests/gap20_quests_entropy.json")
out_dir = Path("sweep_plots")
out_dir.mkdir(exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
with open(data_path) as f:
    entries = [json.loads(line) for line in f]

with open(reference_path) as f:
    reference = json.load(f)

# --------------------------------------------------
# Dataset order
# --------------------------------------------------
categories = list(entries[0]["entropies"].keys())
reference_y = [reference.get(k, None) for k in categories]

# --------------------------------------------------
# Derived complexity = n_max + l_max
# --------------------------------------------------
for e in entries:
    e["complexity"] = e["n_max"] + e["l_max"]

complexity_vals = [e["complexity"] for e in entries]
cmin, cmax = min(complexity_vals), max(complexity_vals)

def marker_size(c, smin=40, smax=100):
    return smin + (c - cmin) / (cmax - cmin) * (smax - smin)

# --------------------------------------------------
# Color mapping by strain
# --------------------------------------------------
strain_vals = [e["strain"] for e in entries]
norm = mcolors.Normalize(vmin=min(strain_vals), vmax=max(strain_vals))
cmap = cm.rainbow

# --------------------------------------------------
# Label helper
# --------------------------------------------------
def curve_label(e):
    return f"n_max={e['n_max']} | l_max={e['l_max']} | strain={e['strain']} | features={e['features']}"

# --------------------------------------------------
# 1) Global plot (all curves)
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for e in sorted(entries, key=lambda x: (x["complexity"], x["strain"])):
    y = [e["entropies"][c] for c in categories]
    color = cmap(norm(e["strain"]))
    size = marker_size(e["complexity"])

    ax.plot(
        categories,
        y,
        marker="o",
        markersize=size / 10,
        linewidth=1.6,
        color=color,
        alpha=0.85,
        label=curve_label(e),
    )

# Reference curve
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

ax.set_xlabel("Dataset")
ax.set_ylabel("Entropy")
ax.set_title("Dataset Entropy (SOAP)")

ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, rotation=30, ha="right")

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

fig.tight_layout()
out_all = out_dir / "soap_entropy_strain_rainbow_all.png"
fig.savefig(out_all, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved → {out_all}")

# --------------------------------------------------
# 2) Grouped plots: same (n_max + l_max), different strain
# --------------------------------------------------
grouped = defaultdict(list)
for e in entries:
    grouped[e["complexity"]].append(e)

for comp, lst in sorted(grouped.items()):
    fig, ax = plt.subplots(figsize=(10, 6))

    for e in sorted(lst, key=lambda x: x["strain"]):
        y = [e["entropies"][c] for c in categories]
        color = cmap(norm(e["strain"]))

        ax.plot(
            categories,
            y,
            marker="o",
            markersize=marker_size(comp) / 10,
            linewidth=1.8,
            color=color,
            alpha=0.9,
            label=curve_label(e)
        )

    # Reference
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

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Entropy")
    ax.set_title(f"Dataset Entropy (SOAP)")

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=30, ha="right")

    ax.legend(
        title="Strain",
        bbox_to_anchor=(1.18, 1.0),
        loc="upper left",
        fontsize=9,
    )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Strain")

    fig.tight_layout()
    out_comp = out_dir / f"soap_entropy_nl_{comp}_strain_rainbow.png"
    fig.savefig(out_comp, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved → {out_comp}")
