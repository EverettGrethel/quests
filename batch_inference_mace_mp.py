"""
MACE-MP inference with embedding extraction from final product layer
"""

import os
os.environ["E3NN_JIT"] = "0"

import argparse
from pathlib import Path

import torch
import numpy as np
from ase.io import read
from mace.calculators import mace_mp, mace_off



def parse_args():
    parser = argparse.ArgumentParser(
        description="MACE-MP inference with embedding extraction"
    )
    parser.add_argument(
        "trajectory_file",
        type=str,
        help="Path to trajectory (.xyz, .extxyz, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mp",
        choices=["mp", "off"],
        help="MACE model type",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="MACE-MP model size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./embeddings",
        help="Directory to save output",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Floating point precision",
    )
    parser.add_argument("--strain", type=float, default=0.0)

    return parser.parse_args()


def build_output_path(trajectory_file: str, model: str, model_size: str, output_dir: str, strain: float):
    dataset_name = Path(trajectory_file).stem
    output_dir = Path(output_dir)
    if strain:
        output_dir = output_dir / f"strain_{strain}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"mace_{model}_{model_size}_{dataset_name}.npz"


def run_inference(
    trajectory_file: str,
    model_name: str,
    model_size: str,
    device: str,
    precision: str,
    strain: float,
):
    dtype = torch.float32 if precision == "float32" else torch.float64

    print(f"Reading trajectory: {trajectory_file}")
    frames = read(trajectory_file, index=":")
    print(f"Found {len(frames)} frames")

    if strain:
        for frame in frames:
            frame.set_cell((1.0 - args.strain) * frame.cell, scale_atoms=True)

    print(f"Loading MACE {model_name} size {model_size}")
    if model_name == "mp":
        calc = mace_mp(
            model=model_size,
            dispersion=False,
            default_dtype=precision,
            device=device,
        )
    elif model_name == "off":
        calc = mace_off(
            model=model_size,
            dispersion=False,
            default_dtype=precision,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = calc.models[0].to(device, dtype=dtype)
    model.eval()

    embeddings_list = []
    energies_list = []
    forces_list = []
    stresses_list = []

    last_embedding = None

    def embedding_hook(module, inputs, output):
        nonlocal last_embedding
        last_embedding = output[0].detach().cpu()

    # hook_handle = model.products[-1].register_forward_hook(embedding_hook)
    hook_handle = model.interactions[-1].register_forward_hook(embedding_hook)

    try:
        for i, atoms in enumerate(frames):
            batch = calc._atoms_to_batch(atoms)
            batch = batch.to(device=device)
            for k in batch.keys:
                v = batch[k]
                if torch.is_tensor(v) and v.is_floating_point():
                    batch[k] = v.to(dtype=dtype)

            out = model(
                batch,
                training=False,
                compute_force=True,
                compute_virials=False,
                compute_stress=True,
                compute_displacement=False,
            )

            energy = out["energy"].item()
            forces = out["forces"].detach().cpu().numpy()
            stresses = out["stress"].detach().cpu().numpy()
            energies_list.append(energy)
            forces_list.append(forces)
            stresses_list.append(stresses)

            if last_embedding is None:
                raise RuntimeError("Embedding hook did not fire")

            embeddings_list.append(last_embedding)

            # print(f"Frame {i + 1}: Energy = {energy:.6f}")

    finally:
        hook_handle.remove()

    # Concatenate atom embeddings across all frames
    all_embeddings = torch.cat(embeddings_list, dim=0)

    return {
        "energy": np.array(energies_list),
        "forces": np.concatenate(forces_list, axis=0),
        "stress": np.concatenate(stresses_list, axis=0),
        "embeddings": all_embeddings.numpy(),
    }


if __name__ == "__main__":
    args = parse_args()

    results = run_inference(
        trajectory_file=args.trajectory_file,
        model_name=args.model,
        model_size=args.model_size,
        device=args.device,
        precision=args.precision,
        strain=args.strain,
    )

    output_file = build_output_path(
        args.trajectory_file,
        args.model,
        args.model_size,
        args.output_dir,
        args.strain,
    )

    print(f"Saving results to: {output_file}")

    np.savez_compressed(
        output_file,
        energy=results["energy"],
        forces=results["forces"],
        stress=results["stress"],
        embeddings=results["embeddings"],
    )

    print("Done.")
