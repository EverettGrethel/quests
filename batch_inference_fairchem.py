"""
Batch inference script with layer embeddings extraction
Extracts node embeddings from the last hidden layer before readout heads
"""

from pathlib import Path
import argparse

from fairchem.core import OCPCalculator
from fairchem.core.preprocessing.atoms_to_graphs import AtomsToGraphs
from fairchem.core.datasets import data_list_collater
from ase.io import read
import torch
import numpy as np


def predict_trajectory_batch(
    trajectory_file: str,
    checkpoint_path: str,
    device: str = "cuda",
    batch_size: int = 1,
    cutoff: float = None,
):
    """
    Predict energy, forces, stress for all frames in a trajectory using batching.
    Extract node embeddings from the last hidden layer.

    Returns:
        Dictionary with 'energy', 'forces', 'stress', 'embeddings'
    """

    print(f"Reading trajectory: {trajectory_file}")
    atoms_list = read(trajectory_file, index=":")
    # from ase.build import bulk

    # atoms = bulk("Cu", "fcc", a=3.61)  # Cu FCC with lattice ~3.61 Å
    # atoms.set_pbc((True, True, True))
    # atoms_list = atoms

    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    n_frames = len(atoms_list)
    print(f"Found {n_frames} frames")

    # Load model
    print("Loading model...")
    calc = OCPCalculator(checkpoint_path=checkpoint_path)
    model = calc.trainer.model.to(device)
    calc.trainer.device = device

    for key in calc.trainer.elementrefs:
        calc.trainer.elementrefs[key] = calc.trainer.elementrefs[key].to(device)

    print(f"Model device: {device}")

    # Override cutoff if provided
    if cutoff is not None:
        print(f"Overriding cutoff to {cutoff} Å")
        calc.trainer.model.cutoff = cutoff

    a2g = AtomsToGraphs(
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_pbc=True,
        r_edges=not calc.trainer.model.otf_graph,
    )

    results = {
        "energy": [],
        "forces": [],
        "stress": [],
        "embeddings": [],
    }

    # Hook for embeddings
    norm_output = None

    def hook_fn(module, input, output):
        nonlocal norm_output
        norm_output = output

    hook = model.backbone.norm.register_forward_hook(hook_fn)

    print(f"Processing {n_frames} frames in batches of {batch_size}...")
    for batch_idx in range(0, n_frames, batch_size):
        batch_frames = atoms_list[batch_idx : batch_idx + batch_size]

        data_list = [a2g.convert(atoms) for atoms in batch_frames]
        batch = data_list_collater(data_list, otf_graph=True).to(device)

        with torch.no_grad():
            predictions = calc.trainer.predict(
                batch, per_image=False, disable_tqdm=True,
            )

        natoms_list = batch.natoms.tolist()

        energies = predictions["energy"]
        forces = torch.split(predictions["forces"], natoms_list)
        stresses = predictions["stress"]

        for i, (energy, force) in enumerate(zip(energies, forces)):
            results["energy"].append(energy.item())
            results["forces"].append(force.cpu().detach().numpy())
            results["stress"].append(stresses[i].cpu().detach().numpy())

        emb_tensor = norm_output.reshape(norm_output.shape[0], -1)
        embeddings_split = torch.split(emb_tensor, natoms_list)

        for emb in embeddings_split:
            results["embeddings"].append(emb.cpu().detach().numpy())

        print(f"  Processed frames {batch_idx + 1}-{batch_idx + len(batch_frames)}")

    hook.remove()

    results["energy"] = np.array(results["energy"])
    results["forces"] = np.concatenate(results["forces"], axis=0)
    results["stress"] = np.concatenate(results["stress"], axis=0)
    results["embeddings"] = np.concatenate(results["embeddings"], axis=0)

    return results


def build_output_path(
    trajectory_file: str,
    checkpoint_path: str,
    output_dir: str,
    save_npz: bool,
    strain: float,
) -> Path:
    """
    Build output filename: <model>_<dataset>.(npz|npy)
    """
    model_name = Path(checkpoint_path).stem
    dataset_name = Path(trajectory_file).stem

    output_dir = Path(output_dir)
    # output_dir = output_dir / ("npz" if save_npz else "npy")
    if strain:
        output_dir = output_dir / f"strain_{strain}"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".npz" if save_npz else ".npy"
    path = output_dir / f"{model_name}_{dataset_name}{suffix}"

    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch inference with embedding extraction"
    )
    parser.add_argument("trajectory_file", type=str, help="Path to trajectory file")
    parser.add_argument("checkpoint_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./embeddings",
        help="Directory to save output",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Override cutoff radius in Angstroms (e.g., 20.0)",
    )
    parser.add_argument("--save_npz", type=int, choices=[0, 1], default=0)
    parser.add_argument("--strain", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    save_npz = bool(args.save_npz)

    results = predict_trajectory_batch(
        trajectory_file=args.trajectory_file,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        batch_size=args.batch_size,
        cutoff=args.cutoff,
    )

    output_file = build_output_path(
        args.trajectory_file, args.checkpoint_path, args.output_dir, save_npz, args.strain,
    )

    print(f"Saving results to: {output_file}")

    if save_npz:
        np.savez_compressed(
            output_file,
            energy=results["energy"],
            forces=results["forces"],
            stress=results["stress"],
            embeddings=results["embeddings"],
        )
    else:
        np.save(output_file, results["embeddings"])

    print("Done.")
