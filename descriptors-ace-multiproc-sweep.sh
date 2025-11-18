#!/usr/bin/env bash
set -euo pipefail

# Make sure the pueue daemon is running before using this:
#   pueued -d
# and optionally set concurrency with:
#   pueue parallel 8

ELEMENTS=(C)
DELTASPLINEBINS="5e-5"
NPOT="FinnisSinclairShiftedScaled"

NDENSITY_LIST=(1)
FS_PARAMETERS_LIST=(
  "1.0 0.5"
)

RCUT_LIST=(5.5 6.5 7.5)
# RCUT_LIST=(5.5)
DCUT_LIST=(0.01 0.2)
# DCUT_LIST=(0.01)
# dummy value, radparameters is rcut
RADPARAM_LIST=(5.5)

NRADS=(
  # "15 12 10 8"
  # "12 10 8 6"
  "10 8 6 4"
  "8 6 6"
  "8 4 2"
  "4"
)
LMAXS=(
  # "0 6 6 5"
  # "0 5 5 4"
  "0 4 4 3"
  "0 3 3"
  "8 6 2"
  "4"
)

DEVICE=(cuda:1 cuda:2 cuda:3)
DATA_PATH="/home/grethel/dev/quests/examples/gap20/{data_name}.xyz"

TEST_SETS=(Graphene Diamond Graphite Nanotubes Fullerenes)
TRAIN_SET="Graphite"
LABELS_PATH="/home/grethel/dev/quests/gap20_quests_entropy.json"
OUTDIR="/home/grethel/dev/quests/sweep_results/sweep_graphite_nradlmax_rcut_dcut.jsonl"

timestamp() { date +"%Y%m%d-%H%M%S"; }

for i in "${!NDENSITY_LIST[@]}"; do
  ndensity="${NDENSITY_LIST[$i]}"
  fs_parameters="${FS_PARAMETERS_LIST[$i]}"

  for rcut in "${RCUT_LIST[@]}"; do
    for dcut in "${DCUT_LIST[@]}"; do
      for radparam in "${RADPARAM_LIST[@]}"; do
        for j in "${!NRADS[@]}"; do
          nrad="${NRADS[$j]}"
          lmax="${LMAXS[$j]}"

          echo "Queuing run: ndensity=$ndensity rcut=$rcut dcut=$dcut nrad=($nrad) lmax=($lmax)"

          pueue add -- python -u descriptors-ace-sweep.py \
            --elements "${ELEMENTS[@]}" \
            --deltaSplineBins "$DELTASPLINEBINS" \
            --npot "$NPOT" \
            --fs_parameters $fs_parameters \
            --ndensity "$ndensity" \
            --radbase "SBessel" \
            --radparameters "$rcut" \
            --rcut "$rcut" \
            --dcut "$dcut" \
            --nrad $nrad \
            --lmax $lmax \
            --device "${DEVICE[@]}" \
            --data_path "$DATA_PATH" \
            --train_set "$TRAIN_SET" \
            --test_sets "${TEST_SETS[@]}" \
            --labels_path "$LABELS_PATH" \
            --out "$OUTDIR"

        done
      done
    done
  done
done

echo "All sweep jobs submitted to pueue."
echo "Use 'pueue status' to monitor and 'pueue parallel N' to set concurrency."
