import os
import fcntl
import time
from contextlib import contextmanager
import argparse
import json
import subprocess
import numpy as np
import torch

from ase.io import read
from quests.gpu.entropy import entropy, entropy_cosine

from gpu_management import exclusive_gpu
from golden_section import optimize_bandwidth_entropy
from embeddings_utils import find_embeddings_file, transform_embeddings


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",             required=True, help="Model name")
    parser.add_argument("--strain",                           type=float, default=0.0)
    parser.add_argument("--cosine",                           type=int, choices=[0, 1], default=0)
    parser.add_argument("--invariant",                        type=int, choices=[0, 1], default=0)
    parser.add_argument("--out",               required=True, help="Output JSONL path")
    parser.add_argument("--device",            required=True, nargs="+", help="One or more CUDA devices, e.g. cuda:0 cuda:1 cuda:2 cuda:3")
    # TODO: make dtype inherit from numpy array
    parser.add_argument("--dtype",                            type=int, choices=[32, 64], default=64)
    parser.add_argument("--min_free_gb",       required=True, type=float, help="Minimum GiB of GPU memory needed to allocate device")
    parser.add_argument("--data_path",         required=True, help="Directory of embeddings files")
    parser.add_argument("--train_set",         required=True, help="Training set for bandwidth tuning")
    parser.add_argument("--test_sets",         required=True, nargs="+", help="List of test sets")
    parser.add_argument("--labels_path",       required=True, help="Path to JSON file with dataset entropy labels")

    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    model = args.model
    out = args.out
    device = args.device
    min_free_gb = args.min_free_gb
    data_path = args.data_path
    train_set = args.train_set
    test_sets = args.test_sets
    print(train_set)
    print(test_sets)
    invariant = bool(args.invariant)
    if invariant:
        print("=========Converting to invariant embeddings=========")
    if int(args.dtype) == 64:
        dtype = torch.float64
        np_dtype = np.float64
    elif int(args.dtype) == 32:
        dtype = torch.float32
        np_dtype = np.float32
    else:
        ValueError(f"{args.dtype} is not a valid dtype.")
    with open(args.labels_path, "r") as f:
        labels = json.load(f)

    # Preliminary check for file existences
    for test_set in test_sets:
        _ = find_embeddings_file(data_path, model, test_set)

    train_set_path = find_embeddings_file(data_path, model, train_set)
    data_train = np.load(train_set_path, allow_pickle=True)
    X_train = np.stack(data_train['embeddings']).astype(np_dtype)
    X_train = transform_embeddings(X_train, model, invariant=invariant)
    print(f"X_train shape {X_train.shape}")

    with exclusive_gpu(device, min_free_gb=min_free_gb, poll_s=30.0) as gpu:
        X_train_tensor = torch.tensor(X_train, device=gpu, dtype=dtype)

        h_opt, opt_report = optimize_bandwidth_entropy(
            X_train_tensor,
            S_star=labels[train_set.split("_")[0]],
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
        del X_train_tensor
        torch.cuda.empty_cache()

    entry = {
        "model": model,
        "train_set": train_set,
        "bandwidth": h_opt,
        "features": X_train.shape[1],
        "strain": args.strain,
        "cosine": bool(args.cosine),
        "entropies": {}
    }

    X_tests_dict = {}
    # TODO: Load test embeddings by model and test_set
    for test_set in test_sets:
        print(test_set)
        if train_set == test_set and not args.strain:
            X_test = X_train
        else:
            test_set_path = find_embeddings_file(data_path, model, test_set)
            if test_set_path is None:
                raise FileNotFoundError(f"Could not find embeddings for model {model} and dataset {test_set} in directory {test_set_path}")
            data_test = np.load(test_set_path, allow_pickle=True)
            X_test = np.stack(data_test['embeddings']).astype(np_dtype)
            X_test = transform_embeddings(X_test, model, invariant=invariant)

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