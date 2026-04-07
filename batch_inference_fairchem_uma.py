"""
UMA / pretrained_mLIP inference with backbone.norm embedding extraction
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from ase.io import read
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.common.distutils import assign_device_for_local_rank


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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--random_weights", type=int, choices=[0, 1], default=0)
    parser.add_argument("--random_seed", type=int, default=None)  # ✅ NEW
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./embeddings",
        help="Directory to save output",
    )
    parser.add_argument("--strain", type=float, default=0.0)
    return parser.parse_args()


# ✅ NEW: robust randomization helper
def randomize_model(model, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # randomize parameters
    for p in model.parameters():
        p.data.copy_(torch.randn_like(p))

    # reset BatchNorm buffers if any
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            if m.running_mean is not None:
                m.running_mean.zero_()
            if m.running_var is not None:
                m.running_var.fill_(1.0)


# ✅ NEW: quick norm diagnostic
def model_param_norm(model):
    total = 0.0
    for p in model.parameters():
        total += p.data.norm().item()
    return total


def build_output_path(
    trajectory_file: str,
    checkpoint_path: str,
    output_dir: str,
    save_npz: bool,
    random_weights: bool,
    strain: float,
) -> Path:
    model_name = Path(checkpoint_path).stem
    dataset_name = Path(trajectory_file).stem

    output_dir = Path(output_dir)
    if strain:
        output_dir = output_dir / f"strain_{strain}"
    # if random_weights:
    #     output_dir = output_dir / "random"
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
    random_seed: int | None,
):
    print(f"Reading trajectory: {trajectory_file}")
    frames = read(trajectory_file, index=":")
    print(f"Found {len(frames)} frames")

    if strain:
        if abs(strain) >= 1.0:
            raise ValueError(f"strain must be < 1.0 and > -1.0, got {strain}")
        for frame in frames:
            frame.set_cell((1.0 - strain) * frame.cell, scale_atoms=True)
    if device == "cpu":
        assign_device_for_local_rank(True, 0)
    else:
        dev, dev_id = device.split(":")
        if dev != "cuda":
            raise ValueError(f"device {device} is not 'cuda:*'")
        assign_device_for_local_rank(False, int(dev_id))
    print(f"Using device {device}")

    print(f"Loading UMA model: {model_name}")
    predictor = pretrained_mlip.get_predict_unit(model_name, device=dev)

    if random_weights:
        before = model_param_norm(predictor.model)
        print(f"Randomizing model weights (seed={random_seed})")
        randomize_model(predictor.model, seed=random_seed)
        after = model_param_norm(predictor.model)
        print(f"Param norm before: {before:.3e}")
        print(f"Param norm after : {after:.3e}")

    calc = FAIRChemCalculator(predictor, task_name="omat")
    model = predictor.model

    embeddings_list = []
    energy_list = []
    forces_list = []
    stress_list = []

    norm_output = None

    def embedding_hook(module, input, output):
        nonlocal norm_output
        norm_output = output.detach().cpu()

    hook_handle = model.module.backbone.norm.register_forward_hook(embedding_hook)

    try:
        for frame in frames:
            frame.calc = calc

            energy_list.append(frame.get_potential_energy())
            forces_list.append(frame.get_forces())
            stress_list.append(frame.get_stress())

            if norm_output is None:
                raise RuntimeError("Embedding hook did not fire")

            embeddings_list.append(norm_output.numpy())

    finally:
        hook_handle.remove()

    if not embeddings_list:
        raise RuntimeError("No embeddings were captured")

    return {
        "energy": np.array(energy_list),
        "forces": np.concatenate(forces_list, axis=0),
        "stress": np.concatenate(stress_list, axis=0),
        "embeddings": np.concatenate(embeddings_list, axis=0),
    }


if __name__ == "__main__":
    args = parse_args()

    results = run_inference(
        trajectory_file=args.trajectory_file,
        model_name=args.model_name,
        device=args.device,
        random_weights=bool(args.random_weights),
        strain=args.strain,
        random_seed=args.random_seed,  # ✅ NEW
    )

    output_file = build_output_path(
        args.trajectory_file,
        args.model_name,
        args.output_dir,
        save_npz=True,
        random_weights=bool(args.random_weights),
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