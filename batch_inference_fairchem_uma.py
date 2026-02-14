"""
UMA / pretrained_mLIP inference with backbone.norm embedding extraction
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from ase.io import read
from fairchem.core import pretrained_mlip, FAIRChemCalculator


def parse_args():
    parser = argparse.ArgumentParser(
        description="UMA inference with embedding extraction"
    )
    parser.add_argument("trajectory_file", type=str, help="Path to trajectory file")
    parser.add_argument(
        "--model_name",
        type=str,
        default="uma-s-1p1",
        help="Pretrained UMA model name",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--random_weights", type=int, choices=[0,1])
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./embeddings",
        help="Directory to save output",
    )
    parser.add_argument("--strain", type=float, default=0.0)
    return parser.parse_args()


def build_output_path(
    trajectory_file: str,
    checkpoint_path: str,
    output_dir: str,
    save_npz: bool,
    random_weights: bool,
    strain: float,
) -> Path:
    """
    Build output filename: <model>_<dataset>.(npz|npy)
    """
    model_name = Path(checkpoint_path).stem
    dataset_name = Path(trajectory_file).stem

    output_dir = Path(output_dir)
    if strain:
        output_dir = output_dir / f"strain_{strain}"
    if random_weights:
        output_dir = output_dir / "random"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".npz" if save_npz else ".npy"
    path = output_dir / f"{model_name}_{dataset_name}{suffix}"

    return path


def run_inference(
    trajectory_file: str,
    model_name: str,
    device: str,
    random_weights: bool,
    strain: float,
):
    print(f"Reading trajectory: {trajectory_file}")
    frames = read(trajectory_file, index=":")
    print(f"Found {len(frames)} frames")

    if strain:
        for frame in frames:
            frame.set_cell((1.0 - args.strain) * frame.cell, scale_atoms=True)

    print(f"Loading UMA model: {model_name}")
    predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
    if random_weights:
        print("Using random model weights")
        for param in predictor.model.parameters():
            param.data = torch.randn_like(param.data)

    calc = FAIRChemCalculator(predictor, task_name="omat")

    model = predictor.model

    embeddings_list = []
    energies_list = []
    forces_list = []
    stresses_list = []

    norm_output = None

    def embedding_hook(module, input, output):
        nonlocal norm_output
        norm_output = output

    hook_handle = model.module.backbone.norm.register_forward_hook(embedding_hook)

    try:
        for i, frame in enumerate(frames):
            frame.calc = calc

            energy = frame.get_potential_energy()
            forces = frame.get_forces()
            stress = frame.get_stress()
            energies_list.append(energy)
            forces_list.append(forces)
            stresses_list.append(stress)

            if norm_output is not None:
                embeddings_list.append(norm_output.detach().cpu())

            # print(f"Frame {i + 1}: Energy = {energy:.6f}")

    finally:
        hook_handle.remove()

    if not embeddings_list:
        raise RuntimeError("No embeddings were captured")

    # Concatenate atom embeddings across all frames
    all_embeddings = torch.cat(embeddings_list, dim=0)
    # all_embeddings = all_embeddings.flatten(start_dim=1)

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
        model_name=args.model_name,
        device=args.device,
        random_weights=args.random_weights,
        strain=args.strain,
    )

    output_file = build_output_path(
        args.trajectory_file,
        args.model_name,
        args.output_dir,
        save_npz=True,
        random_weights=args.random_weights,
        strain=args.strain,
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
