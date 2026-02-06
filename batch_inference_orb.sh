#!/usr/bin/env bash
set -euo pipefail

#######################################
# CPU / threading configuration
#######################################
num_threads=4

# See for other models: /home/grethel/env/orb_v3/lib/python3.11/site-packages/orb_models/forcefield/pretrained.py
MODELS=(
  orb-v3-conservative-inf-omat
  # orb-v3-conservative-20-omat
  # orb-v3-conservative-inf-mpa
  # orb-v3-conservative-20-mpa
)

DATASETS=(
  Diamond_two_frames
  # Graphene
  # Diamond
  # Graphite
  # Nanotubes
  # Fullerenes
  # Liquid
  # Graphene_reflect_invert
  # Diamond_reflect_invert
  # Graphite_reflect_invert
  # Nanotubes_reflect_invert
  # Fullerenes_reflect_invert
  # Liquid_reflect_invert
)

DATA_PATH_TEMPLATE="/home/grethel/dev/quests/examples/gap20/{dataset}.xyz"
# DATA_PATH_TEMPLATE="/home/grethel/dev/quests/examples/gap20_reflect_invert/{dataset}.xyz"
# OUTDIR="/home/grethel/dev/quests/embeddings/npz"
OUTDIR="/data/grethel/embeddings/reflect_invert/npz"

# Batch size
BATCH_SIZE=20

# Device
DEVICE="cuda:3"

# Optional precision for ORB (leave empty to omit)
PRECISION="float64"

# Python executable / env
VENV_PYTHON="/home/grethel/env/orb_v3/bin/python"

#######################################
# Submit sweep
#######################################
job_id=0

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    job_id=$((job_id + 1))
    trajectory_file="${DATA_PATH_TEMPLATE/\{dataset\}/$dataset}"

    echo "Queuing:"
    echo "  dataset=$dataset"
    echo "  batch_size=$BATCH_SIZE"
    echo "  device=$DEVICE"
    echo "  trajectory_file=$trajectory_file"

    # Build optional precision args
    precision_args=()
    if [[ -n "${PRECISION}" ]]; then
      precision_args+=( --precision "${PRECISION}" )
    fi

    pueue add --group orb -- \
      NUMEXPR_NUM_THREADS=$num_threads \
      OMP_NUM_THREADS=$num_threads \
      OPENBLAS_NUM_THREADS=$num_threads \
      MKL_NUM_THREADS=$num_threads \
      LOKY_MAX_CPU_COUNT=$num_threads \
      "$VENV_PYTHON" -u batch_inference_orb.py \
        "$trajectory_file" \
        --model "$model" \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTDIR" \
        "${precision_args[@]}"
  done
done

echo
echo "All jobs submitted."
