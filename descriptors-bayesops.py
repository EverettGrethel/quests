import math
import numpy as np
import json

from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement

from ase.build import bulk, make_supercell
from ase import Atoms
from ase.io import read, write

from pyace import create_multispecies_basis_config
from pyace.activelearning import compute_B_projections
from quests.entropy import perfect_entropy

def idx_to_choice(idx_float, choices):
    i = int(round(max(0, min(len(choices)-1, idx_float))))
    return choices[i]

def make_basis_config(params):
    def iround(x, lo, hi):
        return int(max(lo, min(hi, round(x))))

    radbase = idx_to_choice(params["radbase_idx"], RADBASE_CHOICES)
    npot    = idx_to_choice(params["npot_idx"], NPOT_CHOICES)

    ndensity = iround(params.get("ndensity", 1), 1, 3)

    n1 = iround(params["nrad1"], 4, 12)
    n2 = iround(params["nrad2"], 4, 12)
    n3 = iround(params["nrad3"], 2, 10)
    n1, n2, n3 = sorted([n1, n2, n3], reverse=True)

    l1 = iround(params["lmax1"], 0, 2)
    l2 = iround(params["lmax2"], 2, 10); l2 = max(l2, l1)
    l3 = iround(params["lmax3"], 2, 8);  l3 = min(l3, l2)

    base_p0 = float(params["fs_p0"])
    base_p1 = float(params["fs_p1"])
    fs_params = []
    for _ in range(ndensity):
        fs_params.extend([base_p0, base_p1])

    assert len(fs_params) == 2 * ndensity, (
        f"fs_parameters length {len(fs_params)} != 2*ndensity {2*ndensity}"
    )

    basis_config = {
        "deltaSplineBins": float(params["deltaSplineBins"]),
        "elements": ['Au'],

        "embeddings": {
            "ALL": {
                "npot": npot,
                "fs_parameters": fs_params,
                "ndensity": ndensity,
            },
        },

        "bonds": {
            "ALL": {
                "radbase": radbase,
                "radparameters": [ float(params["radparam0"]) ],
                "rcut": float(params["rcut"]),
                "dcut": float(params["dcut"]),
            }
        },

        "functions": {
            "ALL": {
                "nradmax_by_orders": [ n1, n2, n3 ],
                "lmax_by_orders"   : [ l1, l2, l3 ],
            }
        }
    }

    return basis_config


def fcc_strain_heuristic(train_basis_config):
    train_basis = create_multispecies_basis_config(train_basis_config)
    
    a1 = 3.58
    supercell = 1
    fcc1 = bulk('Au', 'fcc', a=a1, cubic=True)

    fcc2 = fcc1.copy()
    fcc2.set_cell(0.99 * fcc2.cell, scale_atoms=True)

    # fcc1 = make_supercell(fcc1, np.eye(3) * supercell)
    # fcc2 = make_supercell(fcc2, np.eye(3) * supercell)

    x1 = compute_B_projections(train_basis, [fcc1])[0]
    x2 = compute_B_projections(train_basis, [fcc2])[0]
    
    distance = np.linalg.norm(x1[0] - x2[0])
    return distance


def user_score(train_basis_config):
    test_basis_config = {
        "deltaSplineBins": 0.001,
        "elements": ['C'],

        "embeddings": {
            "ALL": {
            "npot": 'FinnisSinclairShiftedScaled',
            "fs_parameters": [ 1, 1],
            "ndensity": 1, # ?
            },
        },

        "bonds": {
            "ALL": {
            "radbase": "SBessel",
            "radparameters": [ 5.25 ],
            "rcut": 6,
            "dcut": 0.01,
            }
        },

        "functions": {
            # "number_of_functions_per_element": 1000,
            "ALL": {
                "nradmax_by_orders": [ 8,8,6],
                "lmax_by_orders"   : [ 0,6,4] }
        }
    }

    distance = fcc_strain_heuristic(train_basis_config)
    
    test_basis = create_multispecies_basis_config(test_basis_config)
    x_test = compute_B_projections(test_basis, frames_list)[0]
    H = perfect_entropy(x_test, h=distance, batch_size=10000)
    diff = abs(H - quests_entropy)
    return -diff


def target_fn(**train_params):
    train_config = make_basis_config(train_params)
    return user_score(train_config)

RADBASE_CHOICES = ["SBessel"]
NPOT_CHOICES    = ["FinnisSinclairShiftedScaled"] 

pbounds = {
    "rcut": (4.5, 8.0),
    "dcut": (0.005, 0.20),
    "deltaSplineBins": (5e-5, 5e-3),

    "radparam0": (3.0, 8.0),

    "npot_idx": (0, len(NPOT_CHOICES)-1),
    "fs_p0": (0.5, 3.0),
    "fs_p1": (0.5, 3.0),
    "ndensity": (1, 3),

    "nrad1": (4, 12),
    "nrad2": (4, 12),
    "nrad3": (2, 10),

    "lmax1": (0, 2),
    "lmax2": (2, 10),
    "lmax3": (2, 8),

    "radbase_idx": (0, len(RADBASE_CHOICES)-1),
}

quests_entropies = {
    # "Graphene": 4.245179458166078,
    "Nanotubes": 7.0282707526691715,
    "Diamond": 4.318381910272737,
    "Graphite": 5.6085074467370095,
    "Fullerenes": 8.67911004440742,
    "Defects": 9.531933892473086,
    "Surfaces": 9.823139796211985,
    "Liquid": 11.61485589283075,
    "Amorphous_Bulk": 12.183809856122803,
}
# quests_entropies = {
#     "Graphene": 4.245179458166078,
# }

for data_name in quests_entropies.keys():
    print(f"Tuning for {data_name}")
    quests_entropy = quests_entropies[data_name]
    path = f"/home/grethel/dev/quests/examples/gap20/{data_name}.xyz"
    frames_list = read(path, index=":")
    
    acq = ExpectedImprovement(xi=0.01)
   
    optimizer = BayesianOptimization(
        f=target_fn,
        pbounds=pbounds,
        acquisition_function=acq,
        verbose=2,
        random_state=42,
    )
    
    optimizer.maximize(init_points=10, n_iter=40)
    
    best = optimizer.max
    print("Best score & params:")
    print(best)
    
    best_basis_config = make_basis_config(best["params"])
    print(best_basis_config)
    
    with open(f"ace_configs/{data_name}_basis_config.json", "w") as f:
        json.dump(best_basis_config, f, indent=2)

    bandwidth = fcc_strain_heuristic(best_basis_config)
    print(f"{data_name} bandwidth: {bandwidth}")

    best_score = user_score(best_basis_config)
    print(f"{data_name} entropy: {-best_score + quests_entropy}")
