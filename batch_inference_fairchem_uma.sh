#!/usr/bin/env bash
set -euo pipefail

num_threads=16

export NUMEXPR_NUM_THREADS=$num_threads
export OMP_NUM_THREADS=$num_threads
export OPENBLAS_NUM_THREADS=$num_threads
export MKL_NUM_THREADS=$num_threads
export LOKY_MAX_CPU_COUNT=$num_threads

MODELS=(
  uma-s-1p1
  uma-m-1p1
)

DATASETS=(
  # Diamond_two_frames
  # Graphene
  # Diamond
  # Graphite
  # Nanotubes
  # Fullerenes
  # Liquid
  Graphene_reflect_invert
  Diamond_reflect_invert
  Graphite_reflect_invert
  Nanotubes_reflect_invert
  Fullerenes_reflect_invert
  Liquid_reflect_invert
)

# DATA_PATH_TEMPLATE="/home/grethel/dev/quests/examples/gap20/{dataset}.xyz"
DATA_PATH_TEMPLATE="/home/grethel/dev/quests/examples/gap20_reflect_invert/{dataset}.xyz"
# OUTDIR="/data/grethel/embeddings/embeddings_raw/npz"
OUTDIR="/data/grethel/embeddings/reflect_invert/npz"

DEVICE="cuda"

RANDOM_WEIGHTS="1"

#######################################
# Submit sweep
#######################################

job_id=0

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    job_id=$((job_id + 1))

    trajectory_file="${DATA_PATH_TEMPLATE/\{dataset\}/$dataset}"

    echo "Queuing: model=$model dataset=$dataset device=$DEVICE"

    VENV_PYTHON="/home/grethel/env/fairchem/bin/python"

    pueue add --group uma -- \
      "$VENV_PYTHON" -u batch_inference_fairchem_uma.py \
        "$trajectory_file" \
        --model_name "$model" \
        --device "$DEVICE" \
        --random_weights "$RANDOM_WEIGHTS" \
        --output_dir "$OUTDIR"

  done
done

echo
echo "All jobs submitted."
