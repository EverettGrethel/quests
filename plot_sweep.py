# Retry building the rainbow-colored plots.

import json
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
from collections import defaultdict
from pathlib import Path

jsonl_text = """
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [5.5], "rcut": 5.5, "dcut": 0.01}}, "functions": {"ALL": {"nradmax_by_orders": [4], "lmax_by_orders": [4]}}}, "bandwidth": 0.05914239633033193, "entropies": {"Graphene": 2.5614113807678223, "Diamond": 3.336698532104492, "Graphite": 5.608762741088867, "Nanotubes": 5.447670936584473, "Fullerenes": 6.91101598739624}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [5.5], "rcut": 5.5, "dcut": 0.01}}, "functions": {"ALL": {"nradmax_by_orders": [8, 4, 2], "lmax_by_orders": [8, 6, 2]}}}, "bandwidth": 3.3413752706758846, "entropies": {"Graphene": 1.2750449180603027, "Diamond": 4.302067756652832, "Graphite": 5.608306884765625, "Nanotubes": 4.527769088745117, "Fullerenes": 5.766193389892578}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [5.5], "rcut": 5.5, "dcut": 0.2}}, "functions": {"ALL": {"nradmax_by_orders": [4], "lmax_by_orders": [4]}}}, "bandwidth": 0.05914239633033193, "entropies": {"Graphene": 2.5614113807678223, "Diamond": 3.336698532104492, "Graphite": 5.608762741088867, "Nanotubes": 5.447670936584473, "Fullerenes": 6.91101598739624}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [5.5], "rcut": 5.5, "dcut": 0.2}}, "functions": {"ALL": {"nradmax_by_orders": [8, 4, 2], "lmax_by_orders": [8, 6, 2]}}}, "bandwidth": 3.3413752706758846, "entropies": {"Graphene": 1.2750449180603027, "Diamond": 4.302067756652832, "Graphite": 5.608306884765625, "Nanotubes": 4.527769088745117, "Fullerenes": 5.766193389892578}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [5.5], "rcut": 5.5, "dcut": 0.2}}, "functions": {"ALL": {"nradmax_by_orders": [8, 6, 6], "lmax_by_orders": [0, 3, 3]}}}, "bandwidth": 18.972891813932918, "entropies": {"Graphene": 0.4260667860507965, "Diamond": 4.347830772399902, "Graphite": 5.60756778717041, "Nanotubes": 2.6427206993103027, "Fullerenes": 3.4873859882354736}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [5.5], "rcut": 5.5, "dcut": 0.01}}, "functions": {"ALL": {"nradmax_by_orders": [8, 6, 6], "lmax_by_orders": [0, 3, 3]}}}, "bandwidth": 18.972891813932918, "entropies": {"Graphene": 0.4260667860507965, "Diamond": 4.347830772399902, "Graphite": 5.60756778717041, "Nanotubes": 2.6427206993103027, "Fullerenes": 3.4873859882354736}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [6.5], "rcut": 6.5, "dcut": 0.01}}, "functions": {"ALL": {"nradmax_by_orders": [4], "lmax_by_orders": [4]}}}, "bandwidth": 0.06304427201779626, "entropies": {"Graphene": 2.2910990715026855, "Diamond": 3.24444842338562, "Graphite": 5.608877182006836, "Nanotubes": 5.75899076461792, "Fullerenes": 7.19564151763916}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [6.5], "rcut": 6.5, "dcut": 0.01}}, "functions": {"ALL": {"nradmax_by_orders": [8, 4, 2], "lmax_by_orders": [8, 6, 2]}}}, "bandwidth": 5.169922764321865, "entropies": {"Graphene": 0.9023487567901611, "Diamond": 3.8486578464508057, "Graphite": 5.609141826629639, "Nanotubes": 4.266839504241943, "Fullerenes": 5.224856376647949}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [6.5], "rcut": 6.5, "dcut": 0.01}}, "functions": {"ALL": {"nradmax_by_orders": [8, 6, 6], "lmax_by_orders": [0, 3, 3]}}}, "bandwidth": 23.03534535777526, "entropies": {"Graphene": 0.283836305141449, "Diamond": 4.271400451660156, "Graphite": 5.608151435852051, "Nanotubes": 2.734076976776123, "Fullerenes": 3.692706346511841}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [6.5], "rcut": 6.5, "dcut": 0.2}}, "functions": {"ALL": {"nradmax_by_orders": [8, 4, 2], "lmax_by_orders": [8, 6, 2]}}}, "bandwidth": 5.169922764321865, "entropies": {"Graphene": 0.9023487567901611, "Diamond": 3.8486578464508057, "Graphite": 5.609141826629639, "Nanotubes": 4.266839504241943, "Fullerenes": 5.224856376647949}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [6.5], "rcut": 6.5, "dcut": 0.2}}, "functions": {"ALL": {"nradmax_by_orders": [8, 6, 6], "lmax_by_orders": [0, 3, 3]}}}, "bandwidth": 23.03534535777526, "entropies": {"Graphene": 0.283836305141449, "Diamond": 4.271400451660156, "Graphite": 5.608151435852051, "Nanotubes": 2.734076976776123, "Fullerenes": 3.692706346511841}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [6.5], "rcut": 6.5, "dcut": 0.2}}, "functions": {"ALL": {"nradmax_by_orders": [4], "lmax_by_orders": [4]}}}, "bandwidth": 0.06304427201779626, "entropies": {"Graphene": 2.2910990715026855, "Diamond": 3.24444842338562, "Graphite": 5.608877182006836, "Nanotubes": 5.75899076461792, "Fullerenes": 7.19564151763916}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [7.5], "rcut": 7.5, "dcut": 0.01}}, "functions": {"ALL": {"nradmax_by_orders": [8, 6, 6], "lmax_by_orders": [0, 3, 3]}}}, "bandwidth": 27.64999807045041, "entropies": {"Graphene": 0.209328755736351, "Diamond": 4.130043029785156, "Graphite": 5.60775899887085, "Nanotubes": 2.953826904296875, "Fullerenes": 3.9404141902923584}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [7.5], "rcut": 7.5, "dcut": 0.01}}, "functions": {"ALL": {"nradmax_by_orders": [8, 4, 2], "lmax_by_orders": [8, 6, 2]}}}, "bandwidth": 6.862035719029721, "entropies": {"Graphene": 0.748363733291626, "Diamond": 3.3868942260742188, "Graphite": 5.608326435089111, "Nanotubes": 4.322953701019287, "Fullerenes": 5.0424485206604}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [7.5], "rcut": 7.5, "dcut": 0.01}}, "functions": {"ALL": {"nradmax_by_orders": [4], "lmax_by_orders": [4]}}}, "bandwidth": 0.05936294287249199, "entropies": {"Graphene": 2.1445472240448, "Diamond": 3.1281018257141113, "Graphite": 5.608105182647705, "Nanotubes": 6.271948337554932, "Fullerenes": 8.088702201843262}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [5.5], "rcut": 5.5, "dcut": 0.01}}, "functions": {"ALL": {"nradmax_by_orders": [10, 8, 6, 4], "lmax_by_orders": [0, 4, 4, 3]}}}, "bandwidth": 72.46671683871344, "entropies": {"Graphene": 0.3206206262111664, "Diamond": 4.728985786437988, "Graphite": 5.608575344085693, "Nanotubes": 2.4051015377044678, "Fullerenes": 3.3671212196350098}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [5.5], "rcut": 5.5, "dcut": 0.2}}, "functions": {"ALL": {"nradmax_by_orders": [10, 8, 6, 4], "lmax_by_orders": [0, 4, 4, 3]}}}, "bandwidth": 72.46671683871344, "entropies": {"Graphene": 0.3206206262111664, "Diamond": 4.728985786437988, "Graphite": 5.608575344085693, "Nanotubes": 2.4051015377044678, "Fullerenes": 3.3671212196350098}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [6.5], "rcut": 6.5, "dcut": 0.01}}, "functions": {"ALL": {"nradmax_by_orders": [10, 8, 6, 4], "lmax_by_orders": [0, 4, 4, 3]}}}, "bandwidth": 136.21528951235533, "entropies": {"Graphene": 0.1452082395553589, "Diamond": 4.589402675628662, "Graphite": 5.608575344085693, "Nanotubes": 2.1796064376831055, "Fullerenes": 2.725964307785034}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [7.5], "rcut": 7.5, "dcut": 0.2}}, "functions": {"ALL": {"nradmax_by_orders": [8, 4, 2], "lmax_by_orders": [8, 6, 2]}}}, "bandwidth": 6.862035719029721, "entropies": {"Graphene": 0.748363733291626, "Diamond": 3.3868942260742188, "Graphite": 5.608326435089111, "Nanotubes": 4.322953701019287, "Fullerenes": 5.0424485206604}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [7.5], "rcut": 7.5, "dcut": 0.2}}, "functions": {"ALL": {"nradmax_by_orders": [4], "lmax_by_orders": [4]}}}, "bandwidth": 0.05936294287249199, "entropies": {"Graphene": 2.1445472240448, "Diamond": 3.1281018257141113, "Graphite": 5.608105182647705, "Nanotubes": 6.271948337554932, "Fullerenes": 8.088702201843262}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [6.5], "rcut": 6.5, "dcut": 0.2}}, "functions": {"ALL": {"nradmax_by_orders": [10, 8, 6, 4], "lmax_by_orders": [0, 4, 4, 3]}}}, "bandwidth": 136.21528951235533, "entropies": {"Graphene": 0.1452082395553589, "Diamond": 4.589402675628662, "Graphite": 5.608575344085693, "Nanotubes": 2.1796064376831055, "Fullerenes": 2.725964307785034}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [7.5], "rcut": 7.5, "dcut": 0.2}}, "functions": {"ALL": {"nradmax_by_orders": [8, 6, 6], "lmax_by_orders": [0, 3, 3]}}}, "bandwidth": 27.64999807045041, "entropies": {"Graphene": 0.209328755736351, "Diamond": 4.130043029785156, "Graphite": 5.60775899887085, "Nanotubes": 2.953826904296875, "Fullerenes": 3.9404141902923584}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [7.5], "rcut": 7.5, "dcut": 0.01}}, "functions": {"ALL": {"nradmax_by_orders": [10, 8, 6, 4], "lmax_by_orders": [0, 4, 4, 3]}}}, "bandwidth": 185.14777935444513, "entropies": {"Graphene": 0.11033477634191513, "Diamond": 4.978549480438232, "Graphite": 5.608917713165283, "Nanotubes": 2.433257818222046, "Fullerenes": 2.805593729019165}}
{"basis_config": {"deltaSplineBins": 5e-05, "elements": ["C"], "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1.0, 0.5], "ndensity": 1}}, "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [7.5], "rcut": 7.5, "dcut": 0.2}}, "functions": {"ALL": {"nradmax_by_orders": [10, 8, 6, 4], "lmax_by_orders": [0, 4, 4, 3]}}}, "bandwidth": 185.14777935444513, "entropies": {"Graphene": 0.11033477634191513, "Diamond": 4.978549480438232, "Graphite": 5.608917713165283, "Nanotubes": 2.433257818222046, "Fullerenes": 2.805593729019165}}
""".strip()

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

data_path = "/home/grethel/dev/quests/sweep_results/sweep_graphite_nradlmax_rcut_dcut.jsonl"

with open(data_path, 'r') as file:
    entries = [json.loads(line) for line in file]
categories = list(entries[0]["entropies"].keys())
reference_y = [reference[k] for k in categories]   # same category order

def order_metric(e):
    nrad = e["basis_config"]["functions"]["ALL"]["nradmax_by_orders"]
    return sum(nrad)

order_vals = [order_metric(e) for e in entries]
norm = mcolors.Normalize(vmin=min(order_vals), vmax=max(order_vals))
cmap = cm.rainbow

def basis_label(e):
    b = e["basis_config"]
    fn = b["functions"]["ALL"]
    nrad = fn["nradmax_by_orders"]
    lmax = fn["lmax_by_orders"]
    rcut = b["bonds"]["ALL"]["rcut"]
    dcut = b["bonds"]["ALL"]["dcut"]
    return f"rcut={rcut}, dcut={dcut} | nrad={nrad} lmax={lmax}"

# 1) All curves
fig, ax = plt.subplots(figsize=(10, 6))

for e in entries:
    y = [e["entropies"][k] for k in categories]
    color = cmap(norm(order_metric(e)))
    ax.plot(categories, y, marker="o", color=color, label=basis_label(e))

ax.plot(categories, reference_y, color="black", linewidth=2.5, marker="o",
        label="QUESTS", zorder=10)

ax.set_xlabel("Dataset")
ax.set_ylabel("Entropy")
ax.set_title("Dataset entropy")
ax.set_xticklabels(categories, rotation=30, ha="right")
ax.legend(
    title="Config",
    bbox_to_anchor=(1.15, 1),   # push farther right
    loc="upper left",
    borderaxespad=0.
)


# Proper colorbar placement (fixes MatplotlibDeprecationWarning)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # required
cbar = fig.colorbar(sm, ax=ax, pad=0.02)

fig.tight_layout()
out_all = Path("sweep_plots/entropy_all_curves_rainbow.png")
fig.savefig(out_all, dpi=200, bbox_inches="tight")
plt.close(fig)

# 2) Per (rcut, dcut)
grouped = defaultdict(list)
for e in entries:
    b = e["basis_config"]
    key = (b["bonds"]["ALL"]["rcut"], b["bonds"]["ALL"]["dcut"])
    grouped[key].append(e)

out_paths = []
for (rcut, dcut), lst in sorted(grouped.items()):
    fig, ax = plt.subplots(figsize=(10, 6))
    for e in lst:
        y = [e["entropies"][k] for k in categories]
        fn = e["basis_config"]["functions"]["ALL"]
        label = f"nrad={fn['nradmax_by_orders']} | lmax={fn['lmax_by_orders']}"
        color = cmap(norm(order_metric(e)))
        plt.plot(categories, y, marker="o", color=color, label=label)

    ax.plot(categories, reference_y, color="black", linewidth=2.5, marker="o",
        label="QUESTS", zorder=10)
    plt.xlabel("Dataset")
    plt.ylabel("Entropy")
    plt.title(f"Dataset entropy for rcut={rcut}, dcut={dcut}")
    plt.xticks(rotation=30, ha="right")
    # plt.legend(title="nrad / lmax", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.legend(
    title="Config",
    bbox_to_anchor=(1.15, 1),   # push farther right
    loc="upper left",
    borderaxespad=0.
    )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax.plot(categories, y, marker="o", color=color, label=label)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)

    plt.tight_layout()
    path = Path(f"sweep_plots/entropy_rainbow_rcut{rcut}_dcut{dcut}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    out_paths.append(str(path))

[str(out_all)] + out_paths
