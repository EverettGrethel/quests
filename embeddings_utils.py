from pathlib import Path
import torch

def find_embeddings_file(directory, model, dataset):
    directory = Path(directory)

    if directory.exists():
        for file in directory.iterdir():
            if file.is_file():
                name = file.name
                if name.startswith(model) and name.endswith(f"_{dataset}.npz"):
                    return file.resolve()

    raise FileNotFoundError(
        f"Could not find embeddings (.npy or .npz) for model {model} "
        f"and dataset {dataset} in directories {directory}"
    )

def mace_to_invariant(emb: torch.Tensor) -> torch.Tensor:
    """
    Convert a MACE interaction embedding to rotation-invariant form.

    Args:
        emb: Tensor of shape (N, channels, 16)
             where 16 = 1 + 3 + 5 + 7 for l=0,1,2,3.

    Returns:
        inv: Tensor of shape (N, channels * 4)
             containing invariant features per atom.
    """
    # check input shape
    if emb.ndim != 3 or emb.shape[-1] != 16:
        raise ValueError("Expected emb with shape (N, channels, 16)")

    # l=0 scalars are already invariant
    inv_l0 = emb[:, :, 0]  # (N, channels)

    # compute norms of each equivariant block
    inv_l1 = torch.norm(emb[:, :, 1:4], dim=-1)   # (N, channels)
    inv_l2 = torch.norm(emb[:, :, 4:9], dim=-1)   # (N, channels)
    inv_l3 = torch.norm(emb[:, :, 9:16], dim=-1)  # (N, channels)

    # concatenate into final invariant descriptor
    inv = torch.cat([inv_l0, inv_l1, inv_l2, inv_l3], dim=-1)  # (N, channels*4)
    return inv


def eqv2_small_to_invariant(emb: torch.Tensor) -> torch.Tensor:
    """
    Convert an Equiformer v2 equivariant embedding
    of shape (n_atoms, channels, 25) to invariant features
    shape (n_atoms, channels * 5).
    """
    # degree 0 scalar part
    inv_l0 = emb[:, :, 0]

    # norms of degrees l=1..4 blocks
    inv_l1 = torch.norm(emb[:, :, 1:4], dim=-1)
    inv_l2 = torch.norm(emb[:, :, 4:9], dim=-1)
    inv_l3 = torch.norm(emb[:, :, 9:16], dim=-1)
    inv_l4 = torch.norm(emb[:, :, 16:25], dim=-1)

    # concatenate per-channel invariants
    return torch.cat([inv_l0, inv_l1, inv_l2, inv_l3, inv_l4], dim=-1)


def eqv2_large_to_invariant(emb: torch.Tensor) -> torch.Tensor:
    """
    Convert an Equiformer v2 equivariant embedding of shape
    (n_atoms, channels, 49) to invariant features of shape
    (n_atoms, channels*7) = (n_atoms, 896).

    49 = 1 + 3 + 5 + 7 + 9 + 11 + 13 for l = 0..6
    """
    # l = 0 (scalar)
    inv_l0 = emb[:, :, 0]                     # (n_atoms, 128)

    # l = 1..6 (vector/tensor norms)
    inv_l1 = torch.norm(emb[:, :, 1:4],  dim=-1)
    inv_l2 = torch.norm(emb[:, :, 4:9],  dim=-1)
    inv_l3 = torch.norm(emb[:, :, 9:16], dim=-1)
    inv_l4 = torch.norm(emb[:, :, 16:25], dim=-1)
    inv_l5 = torch.norm(emb[:, :, 25:36], dim=-1)
    inv_l6 = torch.norm(emb[:, :, 36:49], dim=-1)

    # concatenate invariants
    return torch.cat(
        [inv_l0, inv_l1, inv_l2, inv_l3, inv_l4, inv_l5, inv_l6],
        dim=-1
    )


def transform_embeddings(X, model, invariant=False):
    if X.ndim == 2:
        print("Embedding is already 2-dimensional.")
        return X
    if model.startswith("uma"):
        return X.reshape(X.shape[0], -1)
    elif model.startswith("eqV2"):
        return X.reshape(-1, X.shape[-1])
    elif model.startswith("mace"):
        if invariant:
            return mace_to_invariant(torch.tensor(X)).numpy()
        else:
            return X.reshape(X.shape[0], -1)
    elif model.startswith("orb"):
        return X.reshape(-1, X.shape[-1])
    else:
        raise ValueError(f"Unknown model {model}")
    