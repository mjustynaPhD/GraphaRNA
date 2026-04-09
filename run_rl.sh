#!/bin/bash
#
# Run RL fine-tuning phase for Kinematic RNA refinement.
#
# Phase 2: RL fine-tuning with PPO
#   - Initialize from supervised checkpoint
#   - Freeze LDDTSuggester, train KinematicGNN via PPO
#   - Reward: R = w_bp·BP_recovery - w_clash·Clashes - w_torsion·Torsion
#   - Curriculum continues: σ increases when rolling BP recovery > 80%
#   - Eventually reaches max difficulty (fold from scratch)
#
# Usage:
#   bash run_rl.sh CHECKPOINT [DATASET] [EPOCHS] [LR] [BATCH_SIZE]
#
# Example:
#   bash run_rl.sh save/kinematic_supervised/supervised_epoch_100.pt
#   bash run_rl.sh save/kinematic_supervised/supervised_epoch_100.pt full-3d 50 3e-4 4
#   GPU=1 bash run_rl.sh save/exp/supervised_epoch_150.pt P-only 100 1e-4 8

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

if [ -z "$1" ]; then
    echo "Error: Checkpoint path required!"
    echo "Usage: bash run_rl.sh CHECKPOINT [DATASET] [EPOCHS] [LR] [BATCH_SIZE]"
    echo ""
    echo "Example:"
    echo "  bash run_rl.sh save/kinematic_supervised/supervised_epoch_100.pt"
    exit 1
fi

CHECKPOINT="$1"
DATASET="${2:-full-3d}"
EPOCHS="${3:-50}"
LR="${4:-3e-4}"
BATCH_SIZE="${5:-4}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
EXP_NAME="${EXP_NAME:-kinematic_rl}"

# Model architecture (must match supervised checkpoint)
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
echo "  GraphaRNA - RL Fine-tuning (PPO + Kinematic Refinement)"
echo "========================================================================"
echo "Checkpoint:        $CHECKPOINT"
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

# Check checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# ============================================================================
# Run RL training
# ============================================================================

python -m grapharna.kinematic.main \
    --phase rl \
    --dataset "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --rl-epochs "$EPOCHS" \
    --rl-lr "$LR" \
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
echo "  RL fine-tuning complete!"
echo "========================================================================"
echo "Checkpoints saved in: save/$EXP_NAME/"
echo "Latest checkpoint:    save/$EXP_NAME/rl_epoch_$EPOCHS.pt"
echo ""
echo "Next steps:"
echo "  - Use this checkpoint for sampling: bash run_sample.sh save/$EXP_NAME/rl_epoch_$EPOCHS.pt"
echo ""
