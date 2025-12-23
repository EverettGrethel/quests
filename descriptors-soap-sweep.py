import os
import fcntl
import argparse
import json
import numpy as np
import torch

from ase.io import read
from ase.build import bulk, make_supercell
from dscribe.descriptors import SOAP

from quests.gpu.entropy import entropy, entropy_cosine
from gpu_management import exclusive_gpu
from golden_section import optimize_bandwidth_entropy

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--species", required=True, nargs="+", help="Element symbols")
    parser.add_argument("--r_cut", required=True, type=float)
    parser.add_argument("--n_max", required=True, type=int)
    parser.add_argument("--l_max", required=True, type=int)
    parser.add_argument("--periodic", required=True, type=int, choices=[0, 1])
    parser.add_argument("--strain", required=True, type=float, default=0.0)
    parser.add_argument("--cosine", required=True, type=int, choices=[0, 1])
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", required=True, nargs="+")
    parser.add_argument("--min_free_gb", required=True, type=float)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--train_set", required=True)
    parser.add_argument("--test_sets", required=True, nargs="+")
    parser.add_argument("--labels_path", required=True)

    return parser.parse_args()

def main():
    args = parse_args()
    print(args)

    out = args.out
    device = args.device
    min_free_gb = args.min_free_gb
    data_path = args.data_path
    train_set = args.train_set
    test_sets = args.test_sets

    with open(args.labels_path, "r") as f:
        labels = json.load(f)

    print(f"l_max {args.l_max} n_max {args.n_max}")

    # ---- SOAP INSTANCE ----
    soap = SOAP(
        species=args.species,
        r_cut=args.r_cut,
        n_max=args.n_max,
        l_max=args.l_max,
        periodic=bool(args.periodic),
        sparse=False,
    )

    # ---- LOAD TRAIN FRAMES ----
    train_frames_list = read(data_path.format(data_name=train_set), index=":")
    X_train = soap.create(train_frames_list)
    X_train = np.concatenate(X_train)
    print(f"X_train shape {X_train.shape}")

    # ---- BANDWIDTH OPT ----
    with exclusive_gpu(device, min_free_gb=min_free_gb, poll_s=30.0) as gpu:
        X_train_tensor = torch.tensor(X_train, device=gpu)

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

        del X_train_tensor, train_frames_list
        torch.cuda.empty_cache()

    entry = {
        "descriptor": "SOAP",
        "elements": args.species,
        "r_cut": args.r_cut,
        "n_max": args.n_max,
        "l_max": args.l_max,
        "periodic": bool(args.periodic),
        "strain": args.strain,
        "cosine": bool(args.cosine),
        "bandwidth": h_opt,
        "features": X_train.shape[1],
        "entropies": {}
    }

    # ---- TEST SETS ----
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
            X_test = soap.create(test_frames_list)
            X_test = np.concatenate(X_test)

        print(f"X_test shape {X_test.shape}")
        X_tests_dict[test_set] = X_test

    # ---- ENTROPY EVAL ----
    with exclusive_gpu(device, min_free_gb=min_free_gb, poll_s=30.0) as gpu:
        for test_set in test_sets:
            X_test = X_tests_dict[test_set]
            X_test_tensor = torch.tensor(X_test, device=gpu)

            if not args.cosine:
                S = entropy(X_test_tensor, h=h_opt, batch_size=10000, device=gpu)
            else:
                S = entropy_cosine(X_test_tensor, h=h_opt, batch_size=10000, device=gpu)
            entry["entropies"][test_set] = S.item()

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
