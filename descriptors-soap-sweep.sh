#!/usr/bin/env bash
set -euo pipefail

# Make sure the pueue daemon is running before using this:
#   pueued -d
# and optionally set concurrency with:
#   pueue parallel 8

MIN_FREE_GB="5.0"

SPECIES=(C)

PERIODIC=1

R_CUT_LIST=(5.5 6.5 7.5)
# R_CUT_LIST=(5.5)

# L_MAX_LIST=(8 10 12 15)
# # L_MAX_LIST=(15)
# N_MAX_LIST=(8 10 12 15)
# # N_MAX_LIST=(15)

COMBINATIONS=(
  "8 8"
  "8 10"
  "10 10"
  "10 12"
  "12 12"
  "12 15"
  "15 15"
)

DEVICE=(cuda:0 cuda:1 cuda:2 cuda:3)
# DEVICE=(cuda:2 cuda:3)
# DEVICE=(cuda:2)
DATA_PATH="/home/grethel/dev/quests/examples/gap20/{data_name}.xyz"

TEST_SETS=(Graphene Diamond Graphite Nanotubes Fullerenes Liquid)
TRAIN_SET="Graphene"
# TRAIN_SET="Fullerenes"
LABELS_PATH="/home/grethel/dev/quests/gap20_quests_entropy.json"
OUTDIR="/home/grethel/dev/quests/sweep_results/sweep_soap_graphene.jsonl"

timestamp() { date +"%Y%m%d-%H%M%S"; }

# for r_cut in "${R_CUT_LIST[@]}"; do
#   for l_max in "${L_MAX_LIST[@]}"; do
#     for n_max in "${N_MAX_LIST[@]}"; do

#       echo "Queuing run: r_cut=$r_cut l_max=$l_max n_max=($n_max)"

#       pueue add -- python -u descriptors-soap-sweep.py \
#         --species "${SPECIES[@]}" \
#         --r_cut "$r_cut" \
#         --l_max $l_max \
#         --n_max $n_max \
#         --periodic $PERIODIC \
#         --device "${DEVICE[@]}" \
#         --min_free_gb "$MIN_FREE_GB" \
#         --data_path "$DATA_PATH" \
#         --train_set "$TRAIN_SET" \
#         --test_sets "${TEST_SETS[@]}" \
#         --labels_path "$LABELS_PATH" \
#         --out "$OUTDIR"

#     done
#   done
# done


for r_cut in "${R_CUT_LIST[@]}"; do
  for combo in "${COMBINATIONS[@]}"; do
    read -r l_max n_max <<< "$combo"

    echo "Queuing run: r_cut=$r_cut l_max=$l_max n_max=$n_max"

    pueue add -- python -u descriptors-soap-sweep.py \
      --species "${SPECIES[@]}" \
      --r_cut "$r_cut" \
      --l_max $l_max \
      --n_max $n_max \
      --periodic $PERIODIC \
      --device "${DEVICE[@]}" \
      --min_free_gb "$MIN_FREE_GB" \
      --data_path "$DATA_PATH" \
      --train_set "$TRAIN_SET" \
      --test_sets "${TEST_SETS[@]}" \
      --labels_path "$LABELS_PATH" \
      --out "$OUTDIR"

  done
done

echo "All sweep jobs submitted to pueue."
echo "Use 'pueue status' to monitor and 'pueue parallel N' to set concurrency."
