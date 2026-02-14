#!/usr/bin/env bash
set -euo pipefail

num_threads=4

export NUMEXPR_NUM_THREADS=$num_threads
export OMP_NUM_THREADS=$num_threads
export OPENBLAS_NUM_THREADS=$num_threads
export MKL_NUM_THREADS=$num_threads

# Make sure the pueue daemon is running before using this:
#   pueued -d
# and optionally set concurrency with:
#   pueue parallel 8

MIN_FREE_GB="5.0"

SPECIES=(C)

COSINE=0
DTYPE=64

STRAIN_LIST=(0.0 0.01 0.05 0.1)
# STRAIN_LIST=(0.0)

# CUTOFF_LIST=(5.0 6.0 7.0)
CUTOFF_LIST=(5.0)

# K_LIST=(32 64 128)
K_LIST=(32)

DEVICE=(cuda:0 cuda:1 cuda:2 cuda:3)
# DEVICE=(cuda:2 cuda:3)
# DEVICE=(cuda:2)
DATA_PATH="/home/grethel/dev/quests/examples/gap20/{data_name}.xyz"

TEST_SETS=(Graphene Diamond Graphite Nanotubes Fullerenes Liquid)
TRAIN_SET="Graphite"
# TRAIN_SET="Fullerenes"
LABELS_PATH="/home/grethel/dev/quests/gap20_quests_entropy.json"
OUTDIR="/home/grethel/dev/quests/sweep_results/sweep_quests_${TRAIN_SET}_strain.jsonl"

timestamp() { date +"%Y%m%d-%H%M%S"; }

for strain in "${STRAIN_LIST[@]}"; do
  for k in "${K_LIST[@]}"; do
    for cutoff in "${CUTOFF_LIST[@]}"; do

      echo "Queuing run: k=$k cutoff=$cutoff strain=$strain"

      pueue add --group quests -- python -u descriptors-quests-sweep.py \
        --species "${SPECIES[@]}" \
        --k "$k" \
        --cutoff "$cutoff" \
        --strain "$strain" \
        --cosine "$COSINE" \
        --device "${DEVICE[@]}" \
        --dtype "$DTYPE" \
        --min_free_gb "$MIN_FREE_GB" \
        --data_path "$DATA_PATH" \
        --train_set "$TRAIN_SET" \
        --test_sets "${TEST_SETS[@]}" \
        --labels_path "$LABELS_PATH" \
        --out "$OUTDIR"

    done
  done
done

echo "All sweep jobs submitted to pueue."
echo "Use 'pueue status' to monitor and 'pueue parallel N' to set concurrency."
