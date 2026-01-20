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

# DATASETS=(
#   Graphene
# )
DATASETS=(
  Graphene
  Diamond
  Graphite
  Nanotubes
  Fullerenes
  Liquid
)

DATA_PATH_TEMPLATE="/home/grethel/dev/quests/examples/gap20/{dataset}.xyz"
OUTDIR="/home/grethel/dev/quests/embeddings"

DEVICE="cuda"

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
        --output_dir "$OUTDIR"

  done
done

echo
echo "All jobs submitted."
