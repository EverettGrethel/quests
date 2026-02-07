import json
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
from pathlib import Path
from collections import defaultdict

# -------------------------------
# Reference values
# -------------------------------
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
}

train_set = "Graphite"

MACE_COLORS = {
    "mace_mp_small": "gold",
    "mace_mp_medium": "green",
    "mace_mp_large": "purple",
    "mace_off_small": "gold",
    "mace_off_medium": "green",
    "mace_off_large": "purple",
}

# -------------------------------
# Paths
# -------------------------------
base_path = Path("/home/grethel/dev/quests/sweep_results/embeddings")
out_dir = Path("sweep_plots")
out_dir.mkdir(exist_ok=True)

# split = "*_invariant.jsonl"
split = f"*{train_set}_reflect_invert_invariant.jsonl"

# Split files
files_invariant = sorted(base_path.glob(split))
files_plain = sorted(f for f in base_path.glob("*.jsonl") if "_invariant" not in f.name)

# -------------------------------
# Helper: load entries
# -------------------------------
def load_entries(files):
    entries = []
    for f in files:
        with open(f, "r") as fh:
            for line in fh:
                e = json.loads(line)
                e["_file"] = f.name
                entries.append(e)
    return entries

def filter_by_model(entries, pattern):
    pattern = pattern.lower()
    return [e for e in entries if pattern in e["model"].lower()]

# -------------------------------
# Plotting function
# -------------------------------

def plot_entropy(entries, title, out_name):
    if not entries:
        print(f"No entries for {title}")
        return

    categories = list(entries[0]["entropies"].keys())
    x = [dataset.split("_")[0] for dataset in categories]
    reference_y = [reference.get(k, None) for k in x]

    # Color by features
    feature_vals = [e["features"] for e in entries]
    norm = mcolors.Normalize(vmin=min(feature_vals), vmax=max(feature_vals))
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(10, 6))

    for e in sorted(entries, key=lambda x: x["features"]):
        y = [e["entropies"][k] for k in categories]

        model_name = e["model"].lower()

        # High-contrast fixed colors for mace_mp models
        if "mace_mp_" in model_name or "mace_off_" in model_name:
            color = next(
                (c for k, c in MACE_COLORS.items() if k in model_name),
                "black",  # fallback just in case
            )
        else:
            color = cmap(norm(e["features"]))

        is_uma = "uma" in e["model"].lower()
        linestyle = "--" if is_uma else "-"

        label = f"{e['model']} | feat={e['features']}"
        ax.plot(
            x,
            y,
            marker="o",
            color=color,
            linestyle=linestyle,
            label=label,
        )

    # Reference curve
    ax.plot(
        x,
        reference_y,
        color="black",
        linewidth=2.5,
        marker="o",
        label="QUESTS",
        zorder=10,
    )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Entropy")
    ax.set_title(title)
    ax.set_xticklabels(x, rotation=30, ha="right")

    ax.legend(
        title="Model | Features",
        bbox_to_anchor=(1.15, 1),
        loc="upper left",
    )

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, label="Features")

    fig.tight_layout()
    fig.savefig(out_dir / out_name, dpi=200, bbox_inches="tight")
    plt.close(fig)

# -------------------------------
# Load data
# -------------------------------
entries_invariant = load_entries(files_invariant)
entries_plain = load_entries(files_plain)

entries_invariant_mace_mp = filter_by_model(entries_invariant, "mace_mp_")
entries_plain_mace_mp = filter_by_model(entries_plain, "mace_mp_")

entries_invariant_mace_off = filter_by_model(entries_invariant, "mace_off_")
entries_plain_mace_off = filter_by_model(entries_plain, "mace_off_")

# -------------------------------
# Make plots
# -------------------------------
plot_entropy(
    entries_invariant,
    title="Dataset Entropy (Equivariant Converted to Invariant Embeddings)",
    out_name=f"entropy_{train_set}_reflect_invert_invariant_models.png",
)

plot_entropy(
    entries_plain,
    title="Dataset Entropy (Raw Embeddings)",
    out_name=f"entropy_{train_set}_non_invariant_models.png",
)

plot_entropy(
    entries_invariant_mace_mp,
    title="Dataset Entropy (Invariant Embeddings, mace_mp_* Models)",
    out_name=f"entropy_{train_set}_reflect_invert_invariant_mace_mp.png",
)

plot_entropy(
    entries_plain_mace_mp,
    title="Dataset Entropy (Raw Embeddings, mace_mp_* Models)",
    out_name=f"entropy_{train_set}_non_invariant_mace_mp.png",
)

plot_entropy(
    entries_invariant_mace_off,
    title="Dataset Entropy (Invariant Embeddings, mace_off_* Models)",
    out_name=f"entropy_{train_set}_reflect_invert_invariant_mace_off.png",
)

plot_entropy(
    entries_plain_mace_off,
    title="Dataset Entropy (Raw Embeddings, mace_off_* Models)",
    out_name=f"entropy_{train_set}_non_invariant_mace_off.png",
)