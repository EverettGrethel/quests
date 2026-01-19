"""
Batch inference script for multiple frames in a trajectory
This demonstrates how to get predictions for multiple frames efficiently
using batching instead of a slow per-frame loop.
"""
from pathlib import Path
from fairchem.core import OCPCalculator
from fairchem.core.preprocessing.atoms_to_graphs import AtomsToGraphs
from fairchem.core.datasets import data_list_collater
from ase.io import read
import torch


def predict_trajectory_batch(trajectory_file: str, checkpoint_path: str, batch_size: int = 4):
    """
    Predict energy, forces, stress for all frames in a trajectory using batching.
    
    Args:
        trajectory_file: Path to trajectory file (any format ASE can read)
        checkpoint_path: Path to FAIRChem model checkpoint
        batch_size: Number of frames to process at once (larger = faster, more memory)
    
    Returns:
        Dictionary with 'energy', 'forces', 'stress' arrays per frame
    """
    
    # Read all frames from trajectory
    print(f"Reading trajectory: {trajectory_file}")
    atoms_list = read(trajectory_file, index=":")[0:8]
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    
    n_frames = len(atoms_list)
    print(f"Found {n_frames} frames")
    
    # Initialize model
    print("Loading model...")
    calc = OCPCalculator(checkpoint_path=checkpoint_path)
    
    # Initialize atoms-to-graph converter
    a2g = AtomsToGraphs(
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_pbc=True,
        r_edges=not calc.trainer.model.otf_graph,
    )
    
    # Storage for results
    results = {'energy': [], 'forces': [], 'stress': []}
    
    # Process frames in batches
    print(f"Processing {n_frames} frames in batches of {batch_size}...")
    for batch_idx in range(0, n_frames, batch_size):
        batch_end = min(batch_idx + batch_size, n_frames)
        batch_frames = atoms_list[batch_idx:batch_end]
        
        # Convert to graphs and create batch
        data_list = [a2g.convert(atoms) for atoms in batch_frames]
        batch = data_list_collater(data_list, otf_graph=True)
        
        # Run inference
        with torch.no_grad():
            predictions = calc.trainer.predict(batch, per_image=False, disable_tqdm=True)
        
        # Split results by frame using natoms
        natoms_list = batch.natoms.tolist()
        energies = predictions['energy']
        forces = torch.split(predictions['forces'], natoms_list)
        stresses = predictions['stress']
        
        # Store results per frame
        for i, (energy, force) in enumerate(zip(energies, forces)):
            results['energy'].append(energy.item())
            results['forces'].append(force.detach().cpu().numpy())
            results['stress'].append(stresses[i].detach().cpu().numpy())
        
        print(f"  Processed frames {batch_idx+1}-{batch_end}")
    
    return results


if __name__ == "__main__":
    # Example usage
    trajectory_file = "/home/grethel/dev/quests/examples/gap20/Graphene.xyz"
    checkpoint_path = "/home/grethel/dev/fairchem_checkpoints/eqV2_31M_omat_mp_salex.pt"
    
    results = predict_trajectory_batch(trajectory_file, checkpoint_path, batch_size=4)
    
    print("\n" + "=" * 50)
    print("Results Summary")
    print("=" * 50)
    print(f"Number of frames: {len(results['energy'])}")
    print(f"Energies: {results['energy']}")
    print(f"Number of force arrays: {len(results['forces'])}")
    print(f"Number of stress arrays: {len(results['stress'])}")
