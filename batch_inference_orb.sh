#!/usr/bin/env bash
set -euo pipefail

#######################################
# CPU / threading configuration
#######################################
num_threads=4

# RANDOM_WEIGHTS="1"
RANDOM_WEIGHTS="0"

RANDOM_SEED=0

# See for other models: /home/grethel/env/orb_v3/lib/python3.11/site-packages/orb_models/forcefield/pretrained.py
MODELS=(
  orb-v3-conservative-inf-omat
  # orb-v3-conservative-20-omat
  # orb-v3-conservative-inf-mpa
  # orb-v3-conservative-20-mpa
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

# Batch size
BATCH_SIZE=20

# Device
# DEVICE="cuda"
DEVICE="cuda:0"
# DEVICE="cpu"

PRECISION="float32-highest"
# PRECISION="float64"

# Python executable / env
VENV_PYTHON="/home/grethel/env/orb_v3/bin/python"

#######################################
# Submit sweep
#######################################
job_id=0

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for strain in "${STRAINS[@]}"; do
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
          --strain "$strain" \
          --random_weights "$RANDOM_WEIGHTS" \
          --random_seed "$RANDOM_SEED" \
          "${precision_args[@]}"
    done
  done
done

echo
echo "All jobs submitted."
