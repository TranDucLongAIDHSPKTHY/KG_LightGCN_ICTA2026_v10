#!/usr/bin/env bash
# scripts/run_ablation.sh — v10. Entity ablation A1/A2/A3/A4.
set -euo pipefail

DATASET="${1:-amazon-book}"
SEEDS="42 0 1 2 3"

echo "======================================================"
echo " Entity Ablation | $DATASET | v10"
echo "======================================================"

python scripts/run_ablation.py \
    --dataset "$DATASET" \
    --kg_types "none,category,brand,full" \
    --seeds $SEEDS

echo "Ablation done. Results → results/tables/"
