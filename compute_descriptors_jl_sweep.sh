#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# User sweep definitions
# -------------------------

# ELEMENTS=("C")
# RCUTS=(4.5 5.0 5.5)
# ORDERS=(2 3)
# TOTALDEGREES=(10 15)
# WLS=(1.0 1.5)
# R0S=(2.0 2.5)

ELEMENTS=("C")
RCUTS=(4.5)
ORDERS=(2)
TOTALDEGREES=(10)
WLS=(1.0)
R0S=(2.0)

# -------------------------
# Dataset configuration
# -------------------------
DATASET_DIR="/home/grethel/dev/quests/examples/gap20"
DATASETS=("Graphite" "Graphene" "Diamond")

# -------------------------
# Julia driver script path
# -------------------------
DRIVER="compute_descriptors.jl"

# -------------------------
# Dispatch all combinations
# -------------------------

for elem in "${ELEMENTS[@]}"; do
for rcut in "${RCUTS[@]}"; do
for order in "${ORDERS[@]}"; do
for td in "${TOTALDEGREES[@]}"; do
for wl in "${WLS[@]}"; do
for r0 in "${R0S[@]}"; do
    
    # ID string used for filenames & pueue grouping
    ID="elem-${elem}_rcut-${rcut}_ord-${order}_td-${td}_wL-${wl}_r0-${r0}"

    # Construct dataset list as comma-separated string
    DATASET_CSV=$(IFS=, ; echo "${DATASETS[*]}")

    # Command to run
    CMD="julia --threads=auto $DRIVER \
        --elements $elem \
        --rcut $rcut \
        --order $order \
        --totaldegree $td \
        --wL $wl \
        --r0 $r0 \
        --dataset_dir $DATASET_DIR \
        --datasets $DATASET_CSV \
        --output descriptors_${ID}.npy"

    echo "Dispatching: $ID"
    echo "command $CMD"
    pueue add "$CMD"

done
done
done
done
done
done

echo "All jobs dispatched to pueue."
