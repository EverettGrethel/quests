#!/usr/bin/env bash
set -euo pipefail

num_threads=16

export NUMEXPR_NUM_THREADS=$num_threads
export OMP_NUM_THREADS=$num_threads
export OPENBLAS_NUM_THREADS=$num_threads
export MKL_NUM_THREADS=$num_threads
export LOKY_MAX_CPU_COUNT=$num_threads

# RANDOM_WEIGHTS="1"
RANDOM_WEIGHTS="0"

RANDOM_SEED=0

MODELS=(
  uma-s-1p1
  # uma-m-1p1
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
  # Graphite_reflect_invert
  Nanotubes_reflect_invert
  # Fullerenes_reflect_invert
  # Liquid_reflect_invert
  # Cu_1000K_1bar_biased_reflect_invert
  # Cu_1000K_1bar_unbiased_reflect_invert
  # Graphite_3000K_100GPa_biased_reflect_invert
  # Cu_1000K_1bar_biased_1800
  # Cu_1000K_1bar_biased_1800_reflect_invert
)

STRAINS=(
  # -0.1
  # -0.05
  -0.01
  # -0.005
  # -0.004
  # -0.003
  # -0.002
  # -0.001
  0.0
  # 0.001
  # 0.002
  # 0.003
  # 0.004
  # 0.005
  0.01
  # 0.05
  # 0.1
)

# DATA_PATH_TEMPLATE="/home/grethel/dev/quests/examples/gap20/{dataset}.xyz"
# OUTDIR="/data/grethel/embeddings/embeddings_raw/npz"

DATA_PATH_TEMPLATE="/home/grethel/dev/quests/examples/gap20_reflect_invert/{dataset}.xyz"
OUTDIR="/data/grethel/embeddings/reflect_invert/npz"

# DATA_PATH_TEMPLATE="/home/grethel/dev/quests/examples/gap20_reflect_invert/{dataset}.xyz"
# OUTDIR="/data/grethel/embeddings/reflect_invert/npz/random"

# DATA_PATH_TEMPLATE="/home/grethel/dev/quests/examples/xiangrui_reflect_invert/{dataset}.xyz"
# OUTDIR="/data/grethel/embeddings/reflect_invert/npz/random"

# DATA_PATH_TEMPLATE="/home/grethel/dev/quests/examples/xiangrui/{dataset}.traj"
# OUTDIR="/data/grethel/embeddings/embeddings_raw/npz"

# DATA_PATH_TEMPLATE="/home/grethel/dev/quests/examples/xiangrui_reflect_invert/{dataset}.xyz"
# OUTDIR="/data/grethel/embeddings/reflect_invert/npz"

DEVICE="cuda:0"

#######################################
# Submit sweep
#######################################

job_id=0

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for strain in "${STRAINS[@]}"; do
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
          --random_seed "$RANDOM_SEED" \
          --output_dir "$OUTDIR" \
          --strain "$strain"
    done
  done
done

echo
echo "All jobs submitted."
