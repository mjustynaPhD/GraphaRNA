#!/bin/bash
#
# Run supervised pre-training phase for Kinematic RNA refinement.
#
# Phase 1: Supervised pre-training
#   - Train KinematicGNN + LDDTSuggester to refine structures
#   - Curriculum noise schedule: σ starts at 0.05 rad (~3°) on SO(3)
#   - When BP recovery > 80%, σ increases adaptively
#   - Loss: FAPE-style frame alignment + pLDDT prediction
#
# Usage:
#   bash run_supervised.sh [DATASET] [EPOCHS] [LR] [BATCH_SIZE]
#
# Example:
#   bash run_supervised.sh full-3d 100 1e-3 4
#   GPU=1 bash run_supervised.sh P-only 150 5e-4 8

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

DATASET="${1:-full-3d}"
EPOCHS="${2:-100}"
LR="${3:-1e-3}"
BATCH_SIZE="${4:-4}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
EXP_NAME="${EXP_NAME:-kinematic_supervised}"

# Model architecture
NODE_DIM="${NODE_DIM:-128}"
EDGE_DIM="${EDGE_DIM:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
N_LAYERS="${N_LAYERS:-4}"
N_VECTOR_FEATURES="${N_VECTOR_FEATURES:-8}"
MAX_REFINEMENT_STEPS="${MAX_REFINEMENT_STEPS:-4}"
LEVER_DAMPING="${LEVER_DAMPING:-0.95}"

# Optional WandB logging
WANDB_FLAG=""
if [ "$WANDB" = "1" ] || [ "$WANDB" = "true" ]; then
    WANDB_FLAG="--wandb"
fi

echo "========================================================================"
echo "  GraphaRNA - Supervised Pre-training (Kinematic Refinement)"
echo "========================================================================"
echo "Dataset:           $DATASET"
echo "Epochs:            $EPOCHS"
echo "Learning Rate:     $LR"
echo "Batch Size:        $BATCH_SIZE"
echo "GPU:               $GPU"
echo "Seed:              $SEED"
echo "Experiment Name:   $EXP_NAME"
echo "------------------------------------------------------------------------"
echo "Model Config:"
echo "  Node dim:        $NODE_DIM"
echo "  Edge dim:        $EDGE_DIM"
echo "  Hidden dim:      $HIDDEN_DIM"
echo "  GNN layers:      $N_LAYERS"
echo "  Vector features: $N_VECTOR_FEATURES"
echo "  Refinement steps: $MAX_REFINEMENT_STEPS"
echo "  Lever damping:   $LEVER_DAMPING"
echo "========================================================================"

# ============================================================================
# Run supervised training
# ============================================================================

python -m grapharna.kinematic.main \
    --phase supervised \
    --dataset "$DATASET" \
    --supervised-epochs "$EPOCHS" \
    --supervised-lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --gpu "$GPU" \
    --seed "$SEED" \
    --exp-name "$EXP_NAME" \
    --node-dim "$NODE_DIM" \
    --edge-dim "$EDGE_DIM" \
    --hidden-dim "$HIDDEN_DIM" \
    --n-layers "$N_LAYERS" \
    --n-vector-features "$N_VECTOR_FEATURES" \
    --max-refinement-steps "$MAX_REFINEMENT_STEPS" \
    --lever-damping "$LEVER_DAMPING" \
    $WANDB_FLAG

echo ""
echo "========================================================================"
echo "  Supervised training complete!"
echo "========================================================================"
echo "Checkpoints saved in: save/$EXP_NAME/"
echo "Latest checkpoint:    save/$EXP_NAME/supervised_epoch_$EPOCHS.pt"
echo ""
echo "Next steps:"
echo "  1. Run RL fine-tuning: bash run_rl.sh"
echo "  2. Or use this checkpoint for sampling"
echo ""
