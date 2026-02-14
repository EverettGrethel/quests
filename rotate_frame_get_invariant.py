import numpy as np
from ase.io import iread
from pathlib import Path
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--results_dir", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--data_dir", required=True)
args = parser.parse_args()

dataset = args.dataset
model = args.model
results_dir = args.results_dir
out_dir = args.out_dir
data_dir = args.data_dir

print(f"Processing dataset={dataset}, model={model}")

results_path = Path(results_dir) / f"{model}_{dataset}_reflect_invert.npz"
out_path = Path(out_dir) / f"{model}_{dataset}_reflect_invert_invariant.npz"
data_path = Path(data_dir) / f"{dataset}_reflect_invert.xyz"

# -----------------------------
# Load natoms only (no frame list)
# -----------------------------
natoms = np.fromiter((len(a) for a in iread(data_path, format="extxyz")),
                     dtype=np.int64)
if natoms.size == 0:
    raise ValueError(f"No frames found in {data_path}")

unique_natoms = np.unique(natoms)
print("Unique natoms:", unique_natoms)
print("Number of frames:", natoms.size)

# -----------------------------
# Load embeddings
# -----------------------------
try:
    X = np.load(results_path, allow_pickle=False)["embeddings"]
except Exception:
    print(f"--------Skipping: embeddings for model={model}, dataset={dataset} not found")
    sys.exit()

print("Original X.shape:", X.shape)

# -----------------------------
# Flatten feature dimensions (keep atom axis)
# -----------------------------
if X.ndim > 2:
    X = X.reshape(X.shape[0], -1)
print("Flattened X.shape:", X.shape)

feat = X.shape[1]

# -----------------------------
# Safety check: atoms must match
# -----------------------------
total_atoms = int(natoms.sum())
if X.shape[0] != total_atoms:
    raise ValueError(
        f"Atom mismatch: embeddings have {X.shape[0]} atoms but frames contain {total_atoms}"
    )

# -----------------------------
# Frame -> atom offsets
# -----------------------------
offsets = np.empty(natoms.size + 1, dtype=np.int64)
offsets[0] = 0
np.cumsum(natoms, out=offsets[1:])

# -----------------------------
# Find contiguous blocks of constant natoms (vectorized)
# blocks are [start, end) frame indices
# -----------------------------
change_idx = np.flatnonzero(natoms[1:] != natoms[:-1]) + 1
starts = np.concatenate(([0], change_idx))
ends = np.concatenate((change_idx, [natoms.size]))

# -----------------------------
# Preallocate output
# If every original frame appears as 8 transformed frames, atom count reduces by 8.
# -----------------------------
if total_atoms % 8 != 0:
    raise ValueError(f"Total atoms {total_atoms} not divisible by 8; check your 8x transform assumption")

out_atoms = total_atoms // 8
X_invariant = np.empty((out_atoms, feat), dtype=X.dtype)

out_ptr = 0

# -----------------------------
# Fast symmetry averaging per block
# -----------------------------
for b0, b1 in zip(starts, ends):
    n = int(natoms[b0])
    nframes = int(b1 - b0)

    if nframes % 8 != 0:
        raise ValueError(f"Frames {b0}-{b1} not divisible by 8 (got {nframes})")

    atom_start = int(offsets[b0])
    atom_end = int(offsets[b1])

    # Sanity: should be exactly nframes*n atoms
    expected = nframes * n
    got = atom_end - atom_start
    if got != expected:
        raise RuntimeError(f"Internal error: expected {expected} atoms in block, got {got}")

    block = X[atom_start:atom_end]                 # (nframes*n, feat)
    block = block.reshape(nframes, n, feat)        # (nframes, n, feat)
    block = block.reshape(nframes // 8, 8, n, feat).mean(axis=1)  # (nframes//8, n, feat)

    flat = block.reshape(-1, feat)                 # ( (nframes//8)*n, feat )
    X_invariant[out_ptr:out_ptr + flat.shape[0]] = flat
    out_ptr += flat.shape[0]

if out_ptr != out_atoms:
    raise RuntimeError(f"Output fill mismatch: wrote {out_ptr} atoms, expected {out_atoms}")

print("Invariant embedding shape:", X_invariant.shape)
np.savez(out_path, embeddings=X_invariant)
print("Saved:", out_path)