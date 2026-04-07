#!/usr/bin/env bash
# Run strain distance computation for all parameters/models found in each directory.
# Jobs are submitted to pueue for parallel execution.
#
# Usage:
#   ./sweep_strain_distances.sh [dataset]
#   dataset defaults to "Graphite"
#
# Prerequisites:
#   pueued -d          # start daemon if not running
#   pueue parallel 8   # set concurrency

set -euo pipefail

DATASET="${1:-Graphite}"

ACE_ROOT="/data/grethel/descriptors/ace/npz"
SOAP_ROOT="/data/grethel/descriptors/soap/npz"
EMBED_ROOT="/data/grethel/embeddings/reflect_invert_invariant/npz"

EMBED_MODELS=(
    uma-s-1p1
    orb-v3-conservative-inf-omat
)

METRICS=(l2 cosine mahalanobis)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/grethel/env/fairchem/bin/python"

# 4 parallel jobs × 8 threads = 32 threads total
NUM_THREADS=8
MAHALANOBIS_RCOND=1e-3
pueue parallel 4 --group strain_distances

echo "=== Sweep: dataset=$DATASET ==="

for metric in "${METRICS[@]}"; do
    for f in "$ACE_ROOT"/*_"${DATASET}".npz; do
        fname="$(basename "$f" .npz)"
        [[ "$fname" == *standardized* ]] && continue
        identifier="${fname%_${DATASET}}"
        output="$SCRIPT_DIR/sweep_results/strain_distances/${metric}/ace/${identifier}_${DATASET}.csv"
        pueue add --group strain_distances -- \
            "$PYTHON" -u "$SCRIPT_DIR/compute_strain_distances.py" \
                --root_dir "$ACE_ROOT" \
                --type ace \
                --dataset "$DATASET" \
                --identifier "$identifier" \
                --metric "$metric" \
                --output "$output" \
                --num_threads "$NUM_THREADS" \
                --mahalanobis_rcond "$MAHALANOBIS_RCOND"
    done

    for f in "$SOAP_ROOT"/*_"${DATASET}".npz; do
        fname="$(basename "$f" .npz)"
        [[ "$fname" == *standardized* ]] && continue
        identifier="${fname%_${DATASET}}"
        output="$SCRIPT_DIR/sweep_results/strain_distances/${metric}/soap/${identifier}_${DATASET}.csv"
        pueue add --group strain_distances -- \
            "$PYTHON" -u "$SCRIPT_DIR/compute_strain_distances.py" \
                --root_dir "$SOAP_ROOT" \
                --type soap \
                --dataset "$DATASET" \
                --identifier "$identifier" \
                --metric "$metric" \
                --output "$output" \
                --num_threads "$NUM_THREADS" \
                --mahalanobis_rcond "$MAHALANOBIS_RCOND"
    done

    for model in "${EMBED_MODELS[@]}"; do
        output="$SCRIPT_DIR/sweep_results/strain_distances/${metric}/embeddings/${model}_${DATASET}.csv"
        pueue add --group strain_distances -- \
            "$PYTHON" -u "$SCRIPT_DIR/compute_strain_distances.py" \
                --root_dir "$EMBED_ROOT" \
                --type embeddings \
                --dataset "$DATASET" \
                --identifier "$model" \
                --metric "$metric" \
                --output "$output" \
                --num_threads "$NUM_THREADS" \
                --mahalanobis_rcond "$MAHALANOBIS_RCOND"
    done
done

echo ""
echo "All jobs submitted. Monitor with: pueue status --group strain_distances"
