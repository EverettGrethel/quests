#!/usr/bin/env bash
set -euo pipefail

ELEMENTS='["C"]'
DELTASPLINEBINS="5e-5"
NPOT="FinnisSinclairShiftedScaled"

NDENSITY_LIST=(1 1 1 2)
FS_PARAMETERS_LIST=(
  '[3.0, 0.8]'
  '[1.0, 0.5]'
  '[1.0, 0.8]'
  '[1.0, 1.0, 1.0, 0.5]'
)

# RCUT_LIST=(5.5)
RCUT_LIST=(5.5 6.5 7.5)
# DCUT_LIST=(0.2)
DCUT_LIST=(0.01 0.2)
# RADPARAM_LIST=(5.5)
RADPARAM_LIST=(5.5 6.0)

# NRADS=(
#   '[10, 8, 6, 4]'
# )
# LMAXS=(
#   '[0, 4, 4, 3]'
# )
NRADS=(
  '[10, 8, 6, 4]'
  '[8,6,6]'
  '[8,4,2]'
  '[4]'
)
LMAXS=(
  '[0,4,4,3]'
  '[0,3,3]'
  '[8,6,2]'
  '[4]'
)

DEVICE="cuda:1"
DATA_PATH="/home/grethel/dev/quests/examples/gap20/{data_name}.xyz"
# TESTSETS='["Graphene",
#           "Diamond",
#           "Graphite",
#           "Nanotubes",
#           "Fullerenes",
#           "Defects",
#           "Surfaces",
#           "Liquid",
#           "Amorphous_Bulk"]'
TEST_SETS='["Graphene",
          "Diamond",
          "Graphite",
          "Nanotubes",
          "Fullerenes"]'
TRAIN_SET="Graphite"
LABELS='{"Graphene": 4.245179458166078,
        "Diamond": 4.318381910272738,
        "Graphite": 5.6085074467370095,
        "Nanotubes": 7.0282707526691715,
        "Fullerenes": 8.67911004440742,
        "Defects": 9.531933892473084,
        "Surfaces": 9.823139796211981,
        "Liquid": 11.61485589283075,
        "Amorphous_Bulk": 12.183809856122803}'
OUTDIR="/home/grethel/dev/quests/sweep_results/sweep_graphite.jsonl"

timestamp() { date +"%Y%m%d-%H%M%S"; }

for idx in "${!NDENSITY_LIST[@]}"; do
  ndensity="${NDENSITY_LIST[$idx]}"
  fs_parameters="${FS_PARAMETERS_LIST[$idx]}"
  for rcut in "${RCUT_LIST[@]}"; do
    for dcut in "${DCUT_LIST[@]}"; do
      for radparam in "${RADPARAM_LIST[@]}"; do
        for idx in "${!NRADS[@]}"; do
          nrad="${NRADS[$idx]}"
          lmax="${LMAXS[$idx]}"

          # Make filenames readable
          # nrad_slug="$(sed -e 's/[][]//g' -e 's/,/-/g' -e 's/ //g' <<< "$nrad")"
          # lmax_slug="$(sed -e 's/[][]//g' -e 's/,/-/g' -e 's/ //g' <<< "$lmax")"
          # run_id="nd${ndensity}_rc${rcut}_dc${dcut}_rp${radparam}_nrad${nrad_slug}_lmax${lmax_slug}_$(timestamp)"
          # outfile="${OUTDIR}/${run_id}.jsonl"

          # echo "Running ${run_id}"

          python descriptors-ace-sweep.py \
            --elements "$ELEMENTS" \
            --deltaSplineBins "$DELTASPLINEBINS" \
            --npot "$NPOT" \
            --fs_parameters "$fs_parameters" \
            --ndensity "$ndensity" \
            --radbase "SBessel" \
            --radparameters "[$radparam]" \
            --rcut "$rcut" \
            --dcut "$dcut" \
            --nrad "$nrad" \
            --lmax "$lmax" \
            --device "$DEVICE" \
            --data_path "$DATA_PATH" \
            --train_set "$TRAIN_SET" \
            --test_sets "$TEST_SETS" \
            --labels "$LABELS" \
            --out "$OUTDIR"
        done
      done
    done
  done
done
