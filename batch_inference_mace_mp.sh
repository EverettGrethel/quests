#!/usr/bin/env bash
set -euo pipefail

num_threads=7

PRECISION="float64"

MODELS=(
  mp
  # off
)

MODEL_SIZES=(
  # small
  # medium
  large
)

DATASETS=(
  # Graphene_one_frame
  # Graphene_one_frame_rotated
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
# OUTDIR="/data/grethel/embeddings/embeddings_raw/npz"
OUTDIR="/data/grethel/embeddings/reflect_invert/npz"
DEVICE="cuda:1"

job_id=0

for model in "${MODELS[@]}"; do
  for model_size in "${MODEL_SIZES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      job_id=$((job_id + 1))

      trajectory_file="${DATA_PATH_TEMPLATE/\{dataset\}/$dataset}"

      echo "Queuing: model=$model model_size=$model_size dataset=$dataset device=$DEVICE"

      VENV_PYTHON="/home/grethel/env/mace/bin/python"

      pueue add --group mace -- \
        NUMEXPR_NUM_THREADS=$num_threads \
        OMP_NUM_THREADS=$num_threads \
        OPENBLAS_NUM_THREADS=$num_threads \
        MKL_NUM_THREADS=$num_threads \
        LOKY_MAX_CPU_COUNT=$num_threads \
        "$VENV_PYTHON" -u batch_inference_mace_mp.py \
          "$trajectory_file" \
          --model "$model" \
          --model_size "$model_size" \
          --device "$DEVICE" \
          --precision "$PRECISION" \
          --output_dir "$OUTDIR"

    done
  done
done

echo
echo "All jobs submitted."
