#!/bin/bash
#
# Run full hybrid training: Supervised → RL in one pipeline.
#
# This script runs both phases automatically:
#   Phase 1: Supervised pre-training with curriculum noise (σ=0.05 → 1.5)
#   Phase 2: RL fine-tuning with PPO (continues curriculum)
#
# The curriculum scheduler seamlessly transitions between phases,
# starting from easy (local repair) and progressing to hard (global folding).
#
# Usage:
#   bash run_hybrid.sh [DATASET] [SUP_EPOCHS] [RL_EPOCHS] [BATCH_SIZE]
#
# Example:
#   bash run_hybrid.sh full-3d 100 50 4
#   GPU=1 WANDB=1 bash run_hybrid.sh P-only 150 100 8

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

DATASET="${1:-full-3d}"
SUP_EPOCHS="${2:-100}"
RL_EPOCHS="${3:-50}"
BATCH_SIZE="${4:-4}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
EXP_NAME="${EXP_NAME:-kinematic_hybrid}"

# Learning rates
SUP_LR="${SUP_LR:-1e-3}"
RL_LR="${RL_LR:-3e-4}"

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
echo "  GraphaRNA - Hybrid Training (Supervised + RL)"
echo "========================================================================"
echo "Dataset:              $DATASET"
echo "Supervised Epochs:    $SUP_EPOCHS (LR: $SUP_LR)"
echo "RL Epochs:            $RL_EPOCHS (LR: $RL_LR)"
echo "Batch Size:           $BATCH_SIZE"
echo "GPU:                  $GPU"
echo "Seed:                 $SEED"
echo "Experiment Name:      $EXP_NAME"
echo "------------------------------------------------------------------------"
echo "Model Config:"
echo "  Node dim:           $NODE_DIM"
echo "  Edge dim:           $EDGE_DIM"
echo "  Hidden dim:         $HIDDEN_DIM"
echo "  GNN layers:         $N_LAYERS"
echo "  Vector features:    $N_VECTOR_FEATURES"
echo "  Refinement steps:   $MAX_REFINEMENT_STEPS"
echo "  Lever damping:      $LEVER_DAMPING"
echo "------------------------------------------------------------------------"
echo "Curriculum Noise Schedule:"
echo "  σ_init:             0.05 rad (~3°)"
echo "  σ_max:              1.5 rad (~86°)"
echo "  BP threshold:       80% recovery"
echo "  Growth factor:      1.3×"
echo "  Reward window:      50 steps"
echo "========================================================================"

# ============================================================================
# Run hybrid training
# ============================================================================

python -m grapharna.kinematic.main \
    --phase hybrid \
    --dataset "$DATASET" \
    --supervised-epochs "$SUP_EPOCHS" \
    --rl-epochs "$RL_EPOCHS" \
    --supervised-lr "$SUP_LR" \
    --rl-lr "$RL_LR" \
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
echo "  Hybrid training complete!"
echo "========================================================================"
echo "Checkpoints saved in: save/$EXP_NAME/"
echo "  Supervised:         save/${EXP_NAME}_supervised/supervised_epoch_$SUP_EPOCHS.pt"
echo "  RL:                 save/${EXP_NAME}_rl/rl_epoch_$RL_EPOCHS.pt"
echo ""
echo "Next steps:"
echo "  - Sample structures: bash run_sample.sh save/${EXP_NAME}_rl/rl_epoch_$RL_EPOCHS.pt"
echo ""
