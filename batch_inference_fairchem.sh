#!/usr/bin/env bash
set -euo pipefail

#######################################
# CPU / threading configuration
#######################################

num_threads=4

export NUMEXPR_NUM_THREADS=$num_threads
export OMP_NUM_THREADS=$num_threads
export OPENBLAS_NUM_THREADS=$num_threads
export MKL_NUM_THREADS=$num_threads
export LOKY_MAX_CPU_COUNT=$num_threads

CHECKPOINTS=(
  # /home/grethel/dev/fairchem_checkpoints/eqV2_153M_omat_mp_salex.pt
  # /home/grethel/dev/fairchem_checkpoints/eqV2_86M_omat_mp_salex.pt
  /home/grethel/dev/fairchem_checkpoints/eqV2_dens_31M_mp.pt
  /home/grethel/dev/fairchem_checkpoints/eqV2_31M_omat_mp_salex.pt
  # /home/grethel/dev/fairchem_checkpoints/eqV2_dens_153M_mp.pt
  # /home/grethel/dev/fairchem_checkpoints/eqV2_dens_86M_mp.pt
)

# DATASETS=(Graphene)
DATASETS=(
  Graphene
  Diamond
  Graphite
  Nanotubes
  Fullerenes
  Liquid
)

DATA_PATH_TEMPLATE="/home/grethel/dev/quests/examples/gap20/{dataset}.xyz"

# Batch sizes to sweep
BATCH_SIZE=1

# Devices (round-robin)
DEVICE="cuda"

OUTDIR="/home/grethel/dev/quests/embeddings"

CUTOFF=20

#######################################
# Submit sweep
#######################################

job_id=0

for checkpoint in "${CHECKPOINTS[@]}"; do
  for dataset in "${DATASETS[@]}"; do

    job_id=$((job_id + 1))

    trajectory_file="${DATA_PATH_TEMPLATE/\{dataset\}/$dataset}"

    echo "Queuing:"
    echo "  checkpoint=$(basename "$checkpoint")"
    echo "  dataset=$dataset"
    echo "  batch_size=$BATCH_SIZE"
    echo "  device=$DEVICE"

    VENV_PYTHON="/home/grethel/env/fairchem_1.3.0/bin/python"

    pueue add --group eqv2 -- \
      "$VENV_PYTHON" -u batch_inference_fairchem.py \
        "$trajectory_file" \
        "$checkpoint" \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTDIR"
  done
done

echo
echo "All jobs submitted."
