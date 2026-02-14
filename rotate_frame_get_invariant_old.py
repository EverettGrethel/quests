import numpy as np
from ase.io import iread
from pathlib import Path
import argparse
import sys

# -----------------------------
# Arguments
# -----------------------------
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
# Load frames
# -----------------------------
frames = list(iread(data_path, format="extxyz"))

# -----------------------------
# Load embeddings
# -----------------------------
try:
    X = np.load(results_path, allow_pickle=True)["embeddings"]
except Exception:
    print(f"--------Skipping: embeddings for model={model}, dataset={dataset} not found")
    sys.exit()

print("Original X.shape:", X.shape)

# -----------------------------
# Flatten feature dimensions
# -----------------------------
if X.ndim > 2:
    X = X.reshape(X.shape[0], -1)

print("Flattened X.shape:", X.shape)

# -----------------------------
# Atom count per frame
# -----------------------------
natoms = np.array([len(f) for f in frames])
unique_natoms = np.unique(natoms)

print("Unique natoms:", unique_natoms)
print("Number of frames:", len(frames))

# Safety check
if X.shape[0] != natoms.sum():
    raise ValueError(
        f"Atom mismatch: embeddings have {X.shape[0]} atoms "
        f"but frames contain {natoms.sum()}"
    )

# -----------------------------
# Find contiguous blocks of constant atom count
# -----------------------------
blocks = []
start = 0
for i in range(1, len(natoms)):
    if natoms[i] != natoms[i - 1]:
        blocks.append((start, i, natoms[i - 1]))
        start = i
blocks.append((start, len(natoms), natoms[-1]))

# -----------------------------
# Frame → atom offsets
# -----------------------------
offsets = np.zeros(len(natoms) + 1, dtype=int)
offsets[1:] = np.cumsum(natoms)

# -----------------------------
# Symmetry averaging (groups of 8)
# -----------------------------
avg_embeddings = []

for b0, b1, n in blocks:

    nframes = b1 - b0
    if nframes % 8 != 0:
        raise ValueError(
            f"Frames {b0}-{b1} not divisible by 8 (got {nframes})"
        )

    for i in range(b0, b1, 8):

        # shape: (8, n_atoms, feature_dim)
        sym = np.stack([
            X[offsets[j]:offsets[j + 1]]
            for j in range(i, i + 8)
        ])

        # average over symmetry axis
        avg_embeddings.append(sym.mean(axis=0))

# -----------------------------
# Final invariant embedding
# -----------------------------
X_invariant = np.concatenate(avg_embeddings, axis=0)

print("Invariant embedding shape:", X_invariant.shape)

np.savez(out_path, embeddings=X_invariant)
print("Saved:", out_path)