#!/bin/bash
#
# Quick start script for testing the Kinematic Refinement pipeline.
#
# This script runs a minimal training session for quick validation:
#   - Small dataset subset (if available)
#   - 10 supervised epochs
#   - 5 RL epochs
#   - Small batch size
#
# Perfect for:
#   • Testing the installation
#   • Debugging issues
#   • Quick architecture experiments
#   • CI/CD validation
#
# Usage:
#   bash run_quick_test.sh [DATASET]
#
# Example:
#   bash run_quick_test.sh full-3d
#   GPU=1 bash run_quick_test.sh P-only

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

DATASET="${1:-full-3d}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
EXP_NAME="quick_test_${DATASET}_$(date +%H%M%S)"

# Quick test settings (fast training)
SUP_EPOCHS=10
RL_EPOCHS=5
BATCH_SIZE=2
SUP_LR=1e-3
RL_LR=3e-4

# Smaller model for faster testing
NODE_DIM=64
EDGE_DIM=32
HIDDEN_DIM=64
N_LAYERS=2
N_VECTOR_FEATURES=4
MAX_REFINEMENT_STEPS=2

echo "========================================================================"
echo "  GraphaRNA - Quick Test (Kinematic Refinement)"
echo "========================================================================"
echo "Dataset:              $DATASET"
echo "Supervised Epochs:    $SUP_EPOCHS (fast test mode)"
echo "RL Epochs:            $RL_EPOCHS (fast test mode)"
echo "Batch Size:           $BATCH_SIZE"
echo "GPU:                  $GPU"
echo "Experiment Name:      $EXP_NAME"
echo "------------------------------------------------------------------------"
echo "NOTE: This is a minimal test run for validation only."
echo "      For full training, use run_hybrid.sh or run_full_pipeline.sh"
echo "========================================================================"
echo ""

# ============================================================================
# Run quick test
# ============================================================================

# Export settings as environment variables
export GPU SEED
export NODE_DIM EDGE_DIM HIDDEN_DIM N_LAYERS N_VECTOR_FEATURES
export MAX_REFINEMENT_STEPS SUP_EPOCHS RL_EPOCHS BATCH_SIZE
export SUP_LR RL_LR EXP_NAME

# Run hybrid training (both phases)
bash run_hybrid.sh "$DATASET" "$SUP_EPOCHS" "$RL_EPOCHS" "$BATCH_SIZE"

echo ""
echo "========================================================================"
echo "  ✓ Quick test complete!"
echo "========================================================================"
echo "If you see this message, the pipeline is working correctly."
echo ""
echo "Checkpoints saved in:"
echo "  save/${EXP_NAME}_supervised/"
echo "  save/${EXP_NAME}_rl/"
echo ""
echo "For full training with optimal settings, run:"
echo "  bash run_full_pipeline.sh $DATASET"
echo ""
echo "========================================================================"
