#!/bin/bash
#
# Preprocess RNA dataset to augment with kinematic frame data.
#
# This script augments existing PyG pickled datasets with:
#   - Rigid frames (R, t) computed via Gram-Schmidt from CG coordinates
#   - Local atom coordinates in each frame's reference system
#   - Preprocessed edge information (covalent, BP, spatial)
#
# Usage:
#   bash run_preprocess.sh [DATASET]
#
# Example:
#   bash run_preprocess.sh full-3d
#   bash run_preprocess.sh P-only

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

DATASET="${1:-full-3d}"  # Default to full-3d if not specified
GPU="${GPU:-0}"

echo "========================================================================"
echo "  GraphaRNA - Kinematic Frame Preprocessing"
echo "========================================================================"
echo "Dataset:      $DATASET"
echo "GPU:          $GPU"
echo "========================================================================"

# ============================================================================
# Run preprocessing
# ============================================================================

python -m grapharna.kinematic.main \
    --phase preprocess \
    --dataset "$DATASET" \
    --gpu "$GPU"

echo ""
echo "========================================================================"
echo "  Preprocessing complete!"
echo "========================================================================"
echo "Augmented pickles saved in: data/$DATASET/{train,val,test}-pkl/"
echo ""
