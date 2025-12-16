import os
import numpy as np
import torch
import json
from pathlib import Path
from quests.gpu.entropy import entropy

from golden_section import optimize_bandwidth_entropy

# -------------------------
# Hyperparameter sweep (must match bash script)
# -------------------------
ELEMENTS = ["C"]
RCUTS = [5.5, 6.5, 7.5]
ORDERS = [1, 2, 3]
TOTALDEGREES = [8, 10, 12, 15]
WLS = [1.0]
R0S = [2.0]

TEST_SETS = ["Graphene", "Diamond", "Graphite", "Nanotubes", "Fullerenes", "Liquid"]
TRAIN_SET = "Graphite"

labels_path = "/home/grethel/dev/quests/gap20_quests_entropy.json"

# Path where descriptor npy files were saved
DESCRIPTOR_DIR = Path("/home/grethel/dev/quests/npy_files")

# Output JSONL file
OUTPUT_JSONL = DESCRIPTOR_DIR / "jl_entropy_results.jsonl"

# Hyperparameters for entropy calculation
batch_size = 10000

gpu = "cuda:3"

with open(labels_path, "r") as f:
    labels = json.load(f)

with open(OUTPUT_JSONL, "w") as f_out:
    for elem_list in ELEMENTS:
        elem_tag = elem_list.replace(",", "-")   # safe for filenames
        elem_names = elem_list.split(",")        # true element list

        for rcut in RCUTS:
            for order in ORDERS:
                for td in TOTALDEGREES:
                    for wl in WLS:
                        for r0 in R0S:

                            # Filename-safe ID
                            ID = (
                                f"elements-{elem_tag}_rcut-{rcut}_order-{order}_"
                                f"totaldegree-{td}_wL-{wl}_r0-{r0}"
                            )

                            # Load training descriptors
                            npy_file = DESCRIPTOR_DIR / f"descriptors_{ID}_{TRAIN_SET}.npy"
                            X_train = np.load(npy_file)

                            features = X_train.shape[1]
                            X_train = torch.tensor(X_train, device=gpu)

                            h_opt, opt_report = optimize_bandwidth_entropy(
                                X_train,
                                S_star=labels[TRAIN_SET],
                                batch_size=batch_size,
                                grid_width=100.0,
                                grid_pts=25,
                                device=gpu,
                            )

                            # Compute entropies for all test datasets
                            entropy_results = {}
                            for dataset in TEST_SETS:
                                npy_file = DESCRIPTOR_DIR / f"descriptors_{ID}_{dataset}.npy"
                                if not npy_file.exists():
                                    print(f"Warning: {npy_file} not found, skipping...")
                                    continue

                                X_test = np.load(npy_file)
                                X_test = torch.tensor(X_test, device=gpu)
                                H = entropy(X_test, h=h_opt, batch_size=batch_size)
                                entropy_results[dataset] = float(H)

                            # Output JSON record
                            record = {
                                "elements": elem_names,   # full list of elements
                                "rcut": rcut,
                                "order": order,
                                "totaldegree": td,
                                "wL": wl,
                                "r0": r0,
                                "features": features,
                                "entropy": entropy_results,
                            }

                            f_out.write(json.dumps(record) + "\n")
                            f_out.flush()
                            os.fsync(f_out.fileno())


print(f"Entropy results saved to {OUTPUT_JSONL}")
