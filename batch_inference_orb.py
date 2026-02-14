"""
Batch inference script with layer embeddings extraction (ORB)
Extracts node embeddings from the decoder node_fn input (latent before readout heads)
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math

import numpy as np
import torch
from ase.io import read

from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs


def predict_trajectory_batch(
    trajectory_file: str,
    model: str,
    device: str = "cuda",
    batch_size: int = 20,
    precision: str | None = None,
    strain: float = 0.0,
):
    """
    Predict energy, forces, stress for all frames in a trajectory using batching.
    Extract node embeddings from the last hidden layer before readout heads
    via a forward pre-hook on orbff.model._decoder.node_fn.

    Returns:
        dict with keys: 'energy', 'forces', 'stress', 'embeddings'
            - energy: (n_frames,) float
            - forces: object array of (natoms, 3)
            - stress: object array (shape depends on model output; often (3,3) or (6,))
            - embeddings: object array of (natoms, d_model)
    """
    print(f"Reading trajectory: {trajectory_file}")
    frames = read(trajectory_file, index=":")

    if not isinstance(frames, list):
        frames = [frames]

    n_frames = len(frames)
    print(f"Found {n_frames} frames")

    if strain:
        for frame in frames:
            frame.set_cell((1.0 - args.strain) * frame.cell, scale_atoms=True)

    # Load ORB model
    print("Loading ORB model...")
    orbff = pretrained.ORB_PRETRAINED_MODELS[model](
        device=device,
        **({"precision": precision} if precision is not None else {}),
    )
    print(f"Model device: {device}")

    model_dtype = next(orbff.model.parameters()).dtype
    np_dtype = np.float64 if model_dtype == torch.float64 else np.float32

    results = {
        "energy": [],
        "forces": [],
        "stress": [],
        "embeddings": [],
    }

    # Capture exactly one tensor per forward pass
    captured_batches: list[torch.Tensor] = []

    def prehook_decoder_node_fn(module, inputs):
        # inputs[0] is [N_total_atoms_in_batch, d_model]
        captured_batches.append(inputs[0].detach().cpu())

    hook = orbff.model._decoder.node_fn.register_forward_pre_hook(
        prehook_decoder_node_fn
    )

    try:
        n_batches = math.ceil(n_frames / batch_size)
        print(f"Processing {n_frames} frames in {n_batches} batches of {batch_size}...")

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_frames)
            batch_frames = frames[start:end]

            # Build graphs
            graph_list = [
                atomic_system.ase_atoms_to_atom_graphs(
                    frame, orbff.system_config, device=device
                )
                for frame in batch_frames
            ]
            batched_graph = batch_graphs(graph_list)

            # Clear previous captures and run inference
            captured_batches.clear()
            pred = orbff.predict(batched_graph, split=False)

            if len(captured_batches) != 1:
                raise RuntimeError(
                    f"Expected hook to fire exactly once, got {len(captured_batches)}."
                )

            node_latents = captured_batches[0]  # [sum_atoms_in_batch, d_model] on CPU
            # Split embeddings back per frame
            natoms_list = [int(g.n_node) for g in graph_list]
            embeddings_split = torch.split(node_latents, natoms_list, dim=0)

            # Convert outputs to per-frame using orb helper (this unbatches results)
            atoms_out = atomic_system.atom_graphs_to_ase_atoms(
                batched_graph,
                energy=pred["energy"],
                forces=pred["grad_forces"],
                stress=pred["grad_stress"],
            )

            # Collect results
            for i, atoms in enumerate(atoms_out):
                # Energy: store float
                results["energy"].append(float(atoms.info.get("energy", np.nan)))

                # Forces: natoms x 3
                f = atoms.get_forces()
                results["forces"].append(np.asarray(f, dtype=np_dtype))

                # Stress: might be stored in atoms.info; fallback to atoms.get_stress if available
                s = atoms.info.get("stress", None)
                if s is None:
                    try:
                        s = atoms.get_stress()
                    except Exception:
                        s = None
                results["stress"].append(
                    np.asarray(s, dtype=np_dtype) if s is not None else None
                )

                # Embeddings: natoms x d_model
                emb = embeddings_split[i].numpy()
                results["embeddings"].append(emb)

            print(f"  Processed frames {start + 1}-{end}")

    finally:
        hook.remove()

    # Convert to numpy containers (object arrays for ragged per-frame shapes)
    results["energy"] = np.array(results["energy"])
    results["forces"] = np.concatenate(results["forces"], axis=0)
    results["stress"] = np.concatenate(results["stress"], axis=0)
    results["embeddings"] = np.concatenate(results["embeddings"], axis=0)

    return results


def build_output_path(
    trajectory_file: str,
    model: str,
    output_dir: str,
    save_npz: bool,
    strain: float,
) -> Path:
    """
    Build output filename: <model>_<dataset>.(npz|npy)
    """
    model_name = Path(model).stem
    dataset_name = Path(trajectory_file).stem

    output_dir = Path(output_dir)
    if strain:
        output_dir = output_dir / f"strain_{strain}"
    # output_dir = output_dir / ("npz" if save_npz else "npy")
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".npz" if save_npz else ".npy"
    path = output_dir / f"{model_name}_{dataset_name}{suffix}"

    return path


def parse_args():
    p = argparse.ArgumentParser(description="ORB batch inference with embedding extraction")
    p.add_argument("trajectory_file", type=str, help="Path to trajectory file (.xyz/.extxyz)")
    p.add_argument("--model", type=str, default="orbv3_conservative_inf_omat", help="ORB model name")
    p.add_argument("--device", type=str, default="cuda", help="cuda, cuda:0, cuda:1, or cpu")
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--output_dir", type=str, default="./embeddings_orb", help="Directory to save output")
    p.add_argument(
        "--precision",
        type=str,
        default=None,
        help='Optional ORB precision string (e.g. "float32-high" / "float32-highest" / "float64")',
    )
    p.add_argument("--strain", type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = predict_trajectory_batch(
        trajectory_file=args.trajectory_file,
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        precision=args.precision,
        strain=args.strain,
    )

    output_file = build_output_path(
        trajectory_file=args.trajectory_file,
        model=args.model,
        output_dir=args.output_dir,
        save_npz=True,
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
