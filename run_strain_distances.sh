#!/usr/bin/env bash
# Compute L2 distances between strained and unstrained descriptors/embeddings.
#
# Usage:
#   ./run_strain_distances.sh <type> <dataset> <identifier>
#
# Arguments:
#   type        : ace | soap | embeddings
#   dataset     : e.g. Graphite, Diamond, Graphene, ...
#   identifier  : parameter string for ace/soap (e.g. ace_nrad_10_lmax_10,
#                 soap_nmax_8_lmax_8) or model name for embeddings (e.g. mace_mp_large)
#
# Examples:
#   ./run_strain_distances.sh ace Graphite ace_nrad_10_lmax_10
#   ./run_strain_distances.sh soap Graphite soap_nmax_8_lmax_8
#   ./run_strain_distances.sh embeddings Graphite mace_mp_large

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 <type> <dataset> <identifier> [metric]"
    echo "  type       : ace | soap | embeddings"
    echo "  dataset    : e.g. Graphite"
    echo "  identifier : parameter string (ace/soap) or model name (embeddings)"
    echo "  metric     : l2 | cosine | mahalanobis  (default: l2)"
    exit 1
fi

TYPE="$1"
DATASET="$2"
IDENTIFIER="$3"
METRIC="${4:-l2}"

case "$TYPE" in
    ace)
        ROOT_DIR="/data/grethel/descriptors/ace/npz"
        ;;
    soap)
        ROOT_DIR="/data/grethel/descriptors/soap/npz"
        ;;
    embeddings)
        ROOT_DIR="/data/grethel/embeddings/reflect_invert_invariant/npz"
        ;;
    *)
        echo "ERROR: Unknown type '$TYPE'. Must be one of: ace, soap, embeddings"
        exit 1
        ;;
esac

OUTPUT_DIR="$(dirname "$0")/sweep_results/strain_distances/${METRIC}/${TYPE}"
OUTPUT_FILE="${OUTPUT_DIR}/${IDENTIFIER}_${DATASET}.csv"

echo "Type       : $TYPE"
echo "Dataset    : $DATASET"
echo "Identifier : $IDENTIFIER"
echo "Metric     : $METRIC"
echo "Root dir   : $ROOT_DIR"
echo "Output     : $OUTPUT_FILE"

python3 "$(dirname "$0")/compute_strain_distances.py" \
    --root_dir "$ROOT_DIR" \
    --type "$TYPE" \
    --dataset "$DATASET" \
    --identifier "$IDENTIFIER" \
    --metric "$METRIC" \
    --output "$OUTPUT_FILE"
