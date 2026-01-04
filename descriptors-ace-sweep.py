import os
import fcntl
import time
from contextlib import contextmanager
import argparse
import json
import subprocess
from joblib import Parallel, delayed
import numpy as np
import torch

from ase.io import read
from pyace import create_multispecies_basis_config
from pyace.activelearning import compute_B_projections
from quests.gpu.entropy import entropy, entropy_cosine

from gpu_management import exclusive_gpu
from golden_section import optimize_bandwidth_entropy


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--elements",          required=True, nargs="+", help="Element symbols, e.g. C or C Au")
    parser.add_argument("--deltaSplineBins",   required=True, type=str)
    parser.add_argument("--npot",              required=True, type=str)
    parser.add_argument("--fs_parameters",     required=True, nargs="+", type=float, help="JSON list, e.g. [3.0, 0.8]")
    parser.add_argument("--ndensity",          required=True, type=float)
    parser.add_argument("--radbase",           required=True, type=str)
    parser.add_argument("--radparameters",     required=True, nargs="+", type=float, help="JSON list, e.g. [5.5]")
    parser.add_argument("--rcut",              required=True, type=float)
    parser.add_argument("--dcut",              required=True, type=float)
    parser.add_argument("--nrad",              required=True, nargs="+", type=int, help="JSON list, e.g. [8,4,2]")
    parser.add_argument("--lmax",              required=True, nargs="+", type=int, help="JSON list, e.g. [8,6,2]")
    parser.add_argument("--n_batches",         required=True, type=int)
    parser.add_argument("--strain",            required=True, type=float, default=0.0)
    parser.add_argument("--cosine",            required=True, type=int, choices=[0, 1])
    parser.add_argument("--out",               required=True, help="Output JSONL path")
    parser.add_argument("--device",            required=True, nargs="+", help="One or more CUDA devices, e.g. cuda:0 cuda:1 cuda:2 cuda:3")
    parser.add_argument("--dtype",                            type=int, choices=[32, 64], default=64)
    parser.add_argument("--min_free_gb",       required=True, type=float, help="Minimum GiB of GPU memory needed to allocate device")
    parser.add_argument("--data_path",         required=True, help="f-string path to datasets")
    parser.add_argument("--train_set",         required=True, help="Training set for bandwidth tuning")
    parser.add_argument("--test_sets",         required=True, nargs="+", help="List of test sets")
    parser.add_argument("--labels_path",       required=True, help="Path to JSON file with dataset entropy labels")

    return parser.parse_args()


def make_basis_config(args):
    basis_config = {
        "deltaSplineBins": float(args.deltaSplineBins),
        "elements": args.elements,

        "embeddings": {
            "ALL": {
                "npot": args.npot,
                "fs_parameters": args.fs_parameters,
                "ndensity": int(args.ndensity),
            },
        },

        "bonds": {
            "ALL": {
                "radbase": args.radbase,
                "radparameters": args.radparameters,
                "rcut": args.rcut,
                "dcut": args.dcut,
            }
        },

        "functions": {
            "ALL": {
                "nradmax_by_orders": args.nrad,
                "lmax_by_orders": args.lmax,
            }
        }
    }
    
    return basis_config


def compute_projections(basis, frames, data_type):
    desc_dict = compute_B_projections(basis, frames)
    descriptors = desc_dict[0].astype(data_type)
    return descriptors


def main():
    args = parse_args()
    print(args)
    basis_config = make_basis_config(args)

    out = args.out
    device = args.device
    min_free_gb = args.min_free_gb
    data_path = args.data_path
    train_set = args.train_set
    test_sets = args.test_sets
    print(train_set)
    print(test_sets)
    if int(args.dtype) == 64:
        dtype = torch.float64
        np_dtype = np.float64
    elif int(args.dtype) == 32:
        dtype = torch.float32
        np_dtype = np.float32
    else:
        ValueError(f"{args.dtype} is not a valid dtype.")
    n_batches = args.n_batches
    with open(args.labels_path, "r") as f:
        labels = json.load(f)
    print(f"nrad {args.nrad} lmax {args.lmax}")

    basis = create_multispecies_basis_config(basis_config)
    train_frames_list = read(data_path.format(data_name=train_set), index=":")

    indices = np.array_split(np.arange(len(train_frames_list)), n_batches)
    batches = [[train_frames_list[i] for i in idx] for idx in indices]
    # Uses multi-processing
    X_train = Parallel(n_jobs=n_batches)(
        delayed(compute_projections)(basis, batch, np_dtype) for batch in batches
    )
    X_train = np.concatenate(X_train)
    print(f"X_train shape {X_train.shape}")

    with exclusive_gpu(device, min_free_gb=min_free_gb, poll_s=30.0) as gpu:
        X_train_tensor = torch.tensor(X_train, device=gpu, dtype=dtype)

        h_opt, opt_report = optimize_bandwidth_entropy(
            X_train_tensor,
            S_star=labels[train_set],
            batch_size=10000,
            grid_width=1e5,
            grid_pts=50,
            cosine=args.cosine,
            device=gpu,
        )

        print(f"\n=== Bandwidth optimization ({train_set}) ===")
        print(f"pilot h0         : {opt_report['h0']:.6g}")
        print(f"search log10 span: [{opt_report['log10_bounds'][0]:.3f}, {opt_report['log10_bounds'][1]:.3f}]")
        print(f"best h           : {opt_report['best_h']:.6g}")
        print(f"S(best h)        : {opt_report['best_entropy']:.9f}")
        print(f"target S*        : {opt_report['target_entropy']:.9f}")
        print(f"|S - S*|         : {opt_report['abs_error']:.6g}")

        # Free GPU memory from training
        del X_train_tensor, train_frames_list
        torch.cuda.empty_cache()

    entry = {
        "basis_config": basis_config,
        "bandwidth": h_opt,
        "features": X_train.shape[1],
        "strain": args.strain,
        "cosine": bool(args.cosine),
        "entropies": {}
    }

    X_tests_dict = {}
    for test_set in test_sets:
        print(test_set)
        if train_set == test_set and not args.strain:
            X_test = X_train
        else:
            test_frames_list = read(data_path.format(data_name=test_set), index=":")
            if args.strain:
                for frame in test_frames_list:
                    frame.set_cell((1.0 - args.strain) * frame.cell, scale_atoms=True)
            indices = np.array_split(np.arange(len(test_frames_list)), n_batches)
            batches = [[test_frames_list[i] for i in idx] for idx in indices]
            # Uses multi-processing
            X_test = Parallel(n_jobs=n_batches)(
                delayed(compute_projections)(basis, batch, np_dtype) for batch in batches
            )
            X_test = np.concatenate(X_test)
            del test_frames_list
            
        print(f"X_test shape {X_test.shape}")
        X_tests_dict[test_set] = X_test
        

    with exclusive_gpu(device, min_free_gb=min_free_gb, poll_s=30.0) as gpu:
        for test_set in test_sets:
            X_test = X_tests_dict[test_set]
            X_test_tensor = torch.tensor(X_test, device=gpu)
            if not args.cosine:
                S = entropy(X_test_tensor, h=h_opt, batch_size=10000, device=gpu)
            else:
                S = entropy_cosine(X_test_tensor, h=h_opt, batch_size=10000, device=gpu)
            print(S)
            entry["entropies"][test_set] = S.item()
        # Free test set from GPU memory
        del X_test_tensor
        torch.cuda.empty_cache()

    with open(out, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)
    
    print("Run completed.")

if __name__ == "__main__":
    main()