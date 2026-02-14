#!/usr/bin/env bash
set -euo pipefail

# Make sure the pueue daemon is running:
#   pueued -d
# Set concurrency if desired:
#   pueue parallel 8

############################################
# Global config
############################################

num_threads=4
MIN_FREE_GB="8.0"
DTYPE=64
# DTYPE=32

# CUDA devices passed as a list
DEVICE=(cuda:0 cuda:1 cuda:2 cuda:3)
# DEVICE=(cuda:2 cuda:3)
# DEVICE=(cuda:0)

invariant=0

############################################
# Paths
############################################

LABELS_PATH="/home/grethel/dev/quests/gap20_quests_entropy.json"

# EMBEDDINGS_DIR="/data/grethel/embeddings/reflect_invert_invariant/npz"
# OUTDIR_BASE="/home/grethel/dev/quests/sweep_results/embeddings"

# # ----- Random (UMA) -----
# EMBEDDINGS_DIR="/data/grethel/embeddings/reflect_invert_invariant/npz/random"
# OUTDIR_BASE="/home/grethel/dev/quests/sweep_results/embeddings/random"

# # ----- Strain 0.001 (UMA, MACE) -----
# EMBEDDINGS_DIR="/data/grethel/embeddings/reflect_invert_invariant/npz/strain_0.001"
# OUTDIR_BASE="/home/grethel/dev/quests/sweep_results/embeddings/strain_0.001"

# # ----- Strain 0.01 (UMA, MACE) -----
# EMBEDDINGS_DIR="/data/grethel/embeddings/reflect_invert_invariant/npz/strain_0.01"
# OUTDIR_BASE="/home/grethel/dev/quests/sweep_results/embeddings/strain_0.01"

# # ----- Strain 0.1 (UMA, MACE) -----
# EMBEDDINGS_DIR="/data/grethel/embeddings/reflect_invert_invariant/npz/strain_0.1"
# OUTDIR_BASE="/home/grethel/dev/quests/sweep_results/embeddings/strain_0.1"

############################################
# Sweep parameters
############################################

MODELS=(
  # mace_off_small
  # mace_off_medium
  # mace_off_large
  mace_mp_small
  mace_mp_medium
  mace_mp_large
  # uma-s-1p1
  # uma-m-1p1
  # orb-v3-conservative-inf-omat
  # orb-v3-conservative-20-omat
  # orb-v3-conservative-inf-mpa
  # orb-v3-conservative-20-mpa
  # eqV2_31M_omat_mp_salex
  # eqV2_86M_omat_mp_salex
  # eqV2_153M_omat_mp_salex
  # eqV2_dens_31M_mp
  # eqV2_dens_86M_mp
  # eqV2_dens_153M_mp
)

TRAIN_SETS=(
  # Graphite
  Graphite_reflect_invert_invariant
)

TEST_SETS=(
  # Graphene
  # Diamond
  # Graphite
  # Nanotubes
  # Fullerenes
  # Liquid
  Graphene_reflect_invert_invariant
  Diamond_reflect_invert_invariant
  Graphite_reflect_invert_invariant
  Nanotubes_reflect_invert_invariant
  Fullerenes_reflect_invert_invariant
  Liquid_reflect_invert_invariant
)

############################################
# Helpers
############################################

timestamp() { date +"%Y%m%d-%H%M%S"; }

############################################
# Sweep
############################################

INVARIANT_SUFFIX=""
if [[ "$invariant" -eq 1 ]]; then
  INVARIANT_SUFFIX="_invariant"
fi

for model in "${MODELS[@]}"; do
  for train_set in "${TRAIN_SETS[@]}"; do

    OUTFILE="${OUTDIR_BASE}/sweep_${model}_${train_set}${INVARIANT_SUFFIX}.jsonl"

    echo "Queuing run:"
    echo "  model=$model"
    echo "  train_set=$train_set"

    VENV_PYTHON="/home/grethel/env/quests_py311/bin/python"

    pueue add --group emb -- \
    NUMEXPR_NUM_THREADS=$num_threads \
    OMP_NUM_THREADS=$num_threads \
    OPENBLAS_NUM_THREADS=$num_threads \
    MKL_NUM_THREADS=$num_threads \
    LOKY_MAX_CPU_COUNT=$num_threads \
    "$VENV_PYTHON" -u descriptors-embeddings-sweep.py \
      --model "$model" \
      --device "${DEVICE[@]}" \
      --dtype "$DTYPE" \
      --invariant "$invariant" \
      --min_free_gb "$MIN_FREE_GB" \
      --data_path "$EMBEDDINGS_DIR" \
      --train_set "$train_set" \
      --test_sets "${TEST_SETS[@]}" \
      --labels_path "$LABELS_PATH" \
      --out "$OUTFILE"

  done
done

echo
echo "All entropy sweep jobs submitted to pueue."
echo "Use 'pueue status' to monitor and 'pueue parallel N' to set concurrency."
