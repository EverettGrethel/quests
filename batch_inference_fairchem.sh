#!/usr/bin/env bash
set -euo pipefail

#######################################
# CPU / threading configuration
#######################################

num_threads=4

CHECKPOINTS=(
  /home/grethel/dev/fairchem_checkpoints/eqV2_31M_omat_mp_salex.pt
  /home/grethel/dev/fairchem_checkpoints/eqV2_86M_omat_mp_salex.pt
  /home/grethel/dev/fairchem_checkpoints/eqV2_153M_omat_mp_salex.pt
  /home/grethel/dev/fairchem_checkpoints/eqV2_dens_31M_mp.pt
  /home/grethel/dev/fairchem_checkpoints/eqV2_dens_86M_mp.pt
  /home/grethel/dev/fairchem_checkpoints/eqV2_dens_153M_mp.pt
)

DATASETS=(
  # Graphene_one_frame
  # Graphene_one_frame_rotated
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

STRAINS=(
  0.0
  0.001
  0.01
  0.1
)

# DATA_PATH_TEMPLATE="/home/grethel/dev/quests/examples/gap20/{dataset}.xyz"
DATA_PATH_TEMPLATE="/home/grethel/dev/quests/examples/gap20_reflect_invert/{dataset}.xyz"
# OUTDIR="/data/grethel/embeddings/embeddings_raw/npz"
OUTDIR="/data/grethel/embeddings/reflect_invert/npz"

# Batch sizes to sweep
BATCH_SIZE=1

# Devices (round-robin)
DEVICE="cpu"

CUTOFF=20

save_npz=1

#######################################
# Submit sweep
#######################################

job_id=0

for checkpoint in "${CHECKPOINTS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for strain in "${STRAINS[@]}"; do

      job_id=$((job_id + 1))

      trajectory_file="${DATA_PATH_TEMPLATE/\{dataset\}/$dataset}"

      echo "Queuing:"
      echo "  checkpoint=$(basename "$checkpoint")"
      echo "  dataset=$dataset"
      echo "  batch_size=$BATCH_SIZE"
      echo "  device=$DEVICE"

      VENV_PYTHON="/home/grethel/env/fairchem_1.3.0/bin/python"

      pueue add --group eqv2 -- \
        NUMEXPR_NUM_THREADS=$num_threads \
        OMP_NUM_THREADS=$num_threads \
        OPENBLAS_NUM_THREADS=$num_threads \
        MKL_NUM_THREADS=$num_threads \
        LOKY_MAX_CPU_COUNT=$num_threads \
        "$VENV_PYTHON" -u batch_inference_fairchem.py \
          "$trajectory_file" \
          "$checkpoint" \
          --device "$DEVICE" \
          --batch_size "$BATCH_SIZE" \
          --save_npz "$save_npz" \
          --output_dir "$OUTDIR" \
          --strain "$strain"
    done
  done
done

echo
echo "All jobs submitted."
