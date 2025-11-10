import random
import numpy as np
import json

from ase import Atoms
from ase.io import read, write
from ase.build import bulk, make_supercell

from pyace import create_multispecies_basis_config
from pyace.activelearning import compute_B_projections

from quests.entropy import perfect_entropy, diversity


def make_ace_config(params: dict, elements: list, rcut: float = 5.5, dcut: float = 0.2):
    return {
        "elements": elements,
        "embeddings": {"ALL": {
            "npot": "FinnisSinclairShiftedScaled",
            "fs_parameters": [3.0, 0.8],
            "ndensity": 1,
        }},
        "bonds": {"ALL": {
            "radbase": "SBessel",
            "radparameters": [rcut],
            "rcut": rcut,
            "dcut": dcut,
        }},
        "functions": {"ALL": {
            "nradmax_by_orders": params["nrad"],
            "lmax_by_orders":    params["lmax"],
        }},
        "deltaSplineBins": 5e-5,
    }

def fcc_strain_heuristic(basis, supercell=1):
    a1 = 3.58
    supercell = 1
    fcc1 = bulk('Au', 'fcc', a=a1, cubic=True)

    if supercell > 1:
        # Scale vectors by multiplying with a 3x3 diagonal matrix
        fcc1 = make_supercell(fcc1, np.eye(3) * supercell)

    fcc2 = fcc1.copy()
    fcc2.set_cell(0.99 * fcc2.cell, scale_atoms=True)
    
    x1 = compute_B_projections(basis, [fcc1])[0]
    x2 = compute_B_projections(basis, [fcc2])[0]
    print(x1.shape)
    
    distance = np.linalg.norm(x1[0] - x2[0])
    return distance


def multicomponent_fcc_strain_heuristic(basis, elements=["Au", "Ag"], center_idx=0, frac=1.0, supercell=2, ratio=0.5, seed: int = None):
    fcc1 = bulk(elements[0], "fcc", a=frac * 4.08, cubic=True)
    
    if supercell > 1:
        # Scale vectors by multiplying with a 3x3 diagonal matrix
        fcc1 = make_supercell(fcc1, np.eye(3) * supercell)

    if seed is not None:
        rng = random.Random(seed)

    n_atoms = len(fcc1)
    if center_idx < 0 or center_idx >= n_atoms:
        raise IndexError(f"center_idx {center_idx} out of range for N={n_atoms}.")
    pool = [i for i in range(n_atoms) if i != center_idx]
    # guarantee at least one flip
    k = max(1, int(round(ratio * len(pool)))) if len(pool) > 0 else 0
    chosen = (rng.sample(pool, k) if (rng and k > 0) else (random.sample(pool, k) if k > 0 else []))
    for i in chosen:
        fcc1[i].symbol = elements[1]

    fcc2 = fcc1.copy()
    fcc2.set_cell(0.99 * fcc2.cell, scale_atoms=True)

    basis_elems = set(getattr(basis, "elements", [])) if hasattr(basis, "elements") else None
    if basis_elems is not None and not set(elements).issubset(basis_elems):
        raise ValueError(f"Basis elements {basis_elems} are missing some of {elements}.")
    
    x1 = compute_B_projections(basis, [fcc1])[0]
    x2 = compute_B_projections(basis, [fcc2])[0]
    print(x1.shape)
    
    distance = np.linalg.norm(x1[center_idx] - x2[center_idx])
    return distance


data_names = ["Graphene",
          "Diamond",
          "Graphite",
          "Nanotubes",
          "Fullerenes",
          "Defects",
          "Surfaces",
          "Liquid",
          "Amorphous_Bulk",
         ]
# data_names = ["Graphene"]

sweep = [{"nrad": [4], "lmax": [4]},
         {"nrad": [6, 3], "lmax": [6, 3]},
        {"nrad": [8, 4, 2], "lmax": [8, 6, 2]},
        {"nrad": [10, 6, 3], "lmax": [10, 8, 4]},
        {"nrad": [12, 8, 4], "lmax": [12, 10, 6]},
        ]

seed = 0
supercell = 3
# elements = ["Au", "Ag"]
elements = ["Au"]
# ratios = np.arange(0.1, 1, 0.1)
ratios = np.arange(0.1, 0.2, 0.1)

results_path = "/home/grethel/dev/quests/sweep_results/sweep_fcc.jsonl"

with open(results_path, "w") as f:
    for params in sweep:
        for ratio in ratios:
            ratio = ratio.item()
            print(f"{params}")
            fcc_basis_config = make_ace_config(params=params, elements=elements)
            fcc_basis = create_multispecies_basis_config(fcc_basis_config)
            if len(elements) > 1:
                bandwidth = multicomponent_fcc_strain_heuristic(fcc_basis, elements=elements, seed=seed, supercell=supercell)
            else:
                bandwidth = fcc_strain_heuristic(fcc_basis, supercell=supercell)
            print(f"bandwidth {bandwidth}")
    
            entry = {
                "basis_config": fcc_basis_config,
                "elements": elements,
                "bandwidth": bandwidth,
                "supercell": supercell,
                "ratio": ratio,
                "seed": seed,
                "entropy": {}
            }
            
            for data_name in data_names:
                path = f"/home/grethel/dev/quests/examples/gap20/{data_name}.xyz"
                frames_list = read(path, index=":")
                data_basis_config = make_ace_config(params=params, elements=["C"])
                data_basis = create_multispecies_basis_config(data_basis_config)
                descriptor_ace = compute_B_projections(data_basis, frames_list)[0]
                batch_size = 10000
                H = perfect_entropy(descriptor_ace, h=bandwidth, batch_size=batch_size)
                entry["entropy"][data_name] = float(H)
                print(f"{data_name} entropy: {H}")
        
            f.write(json.dumps(entry) + "\n")
            f.flush()
