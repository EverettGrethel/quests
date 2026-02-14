#!/usr/bin/env bash
set -euo pipefail

#######################################
# CPU / threading configuration
#######################################
# num_threads=4

#######################################
# Paths
#######################################
# RESULTS_DIR="/data/grethel/embeddings/reflect_invert/npz"
# OUT_DIR="/data/grethel/embeddings/reflect_invert_invariant/npz"
# DATA_DIR="/home/grethel/dev/quests/examples/gap20_reflect_invert"

# ----- Random (UMA only) -----
# RESULTS_DIR="/data/grethel/embeddings/reflect_invert/npz/random"
# OUT_DIR="/data/grethel/embeddings/reflect_invert_invariant/npz/random"
# DATA_DIR="/home/grethel/dev/quests/examples/gap20_reflect_invert"

# # ----- Strain 0.001 (UMA, MACE) -----
# RESULTS_DIR="/data/grethel/embeddings/reflect_invert/npz/strain_0.001"
# OUT_DIR="/data/grethel/embeddings/reflect_invert_invariant/npz/strain_0.001"
# DATA_DIR="/home/grethel/dev/quests/examples/gap20_reflect_invert"

# # ----- Strain 0.01 (UMA, MACE) -----
# RESULTS_DIR="/data/grethel/embeddings/reflect_invert/npz/strain_0.01"
# OUT_DIR="/data/grethel/embeddings/reflect_invert_invariant/npz/strain_0.01"
# DATA_DIR="/home/grethel/dev/quests/examples/gap20_reflect_invert"

# # ----- Strain 0.1 (UMA, MACE) -----
# RESULTS_DIR="/data/grethel/embeddings/reflect_invert/npz/strain_0.1"
# OUT_DIR="/data/grethel/embeddings/reflect_invert_invariant/npz/strain_0.1"
# DATA_DIR="/home/grethel/dev/quests/examples/gap20_reflect_invert"

# TEST
# RESULTS_DIR="/data/grethel/embeddings/reflect_invert/npz/strain_0.001"
# OUT_DIR="/data/grethel/embeddings/reflect_invert_invariant/npz/test_fast/new"
# DATA_DIR="/home/grethel/dev/quests/examples/gap20_reflect_invert"

#######################################
# Models & datasets
#######################################
MODELS=(
  mace_mp_small
  mace_mp_medium
  mace_mp_large
  uma-s-1p1
  uma-m-1p1
  # mace_off_small
  # mace_off_medium
  # mace_off_large
  orb-v3-conservative-inf-omat
  orb-v3-conservative-20-omat
  orb-v3-conservative-inf-mpa
  orb-v3-conservative-20-mpa
)

DATASETS=(
  # Graphene
  # Diamond
  # Graphite
  # Nanotubes
  Fullerenes
  Liquid
)

#######################################
# Python executable / env
#######################################
VENV_PYTHON="/home/grethel/env/quests_py311/bin/python"

#######################################
# Script to run
#######################################
SCRIPT="rotate_frame_get_invariant.py"

#######################################
# Submit sweep
#######################################
job_id=0

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    job_id=$((job_id + 1))

    echo "Queuing job $job_id:"
    echo "  dataset     = $dataset"
    echo "  model       = $model"
    echo "  results_dir = $RESULTS_DIR"
    echo "  out_dir     = $OUT_DIR"
    echo "  data_dir    = $DATA_DIR"
    echo

      # NUMEXPR_NUM_THREADS=$num_threads \
      # OMP_NUM_THREADS=$num_threads \
      # OPENBLAS_NUM_THREADS=$num_threads \
      # MKL_NUM_THREADS=$num_threads \
      # LOKY_MAX_CPU_COUNT=$num_threads \
    pueue add --group inv -- \
      "$VENV_PYTHON" -u "$SCRIPT" \
        --dataset "$dataset" \
        --model "$model" \
        --results_dir "$RESULTS_DIR" \
        --out_dir "$OUT_DIR" \
        --data_dir "$DATA_DIR"

  done
done

echo
echo "All invariant-embedding jobs submitted."