#!/bin/bash
#
# Full Pipeline: Preprocessing → Hybrid Training (Supervised + RL)
#
# This master script runs the complete GraphaRNA Kinematic Refinement
# pipeline from start to finish:
#
#   1. Preprocess dataset (augment pickles with frame data)
#   2. Supervised pre-training (100 epochs, curriculum noise σ=0.05→1.5)
#   3. RL fine-tuning (50 epochs, PPO with physics-based reward)
#
# Usage:
#   bash run_full_pipeline.sh [DATASET]
#
# Example:
#   bash run_full_pipeline.sh full-3d
#   GPU=1 WANDB=1 bash run_full_pipeline.sh P-only
#
# Environment variables for customization:
#   GPU              - GPU device ID (default: 0)
#   WANDB            - Enable WandB logging (default: 0)
#   SUP_EPOCHS       - Supervised epochs (default: 100)
#   RL_EPOCHS        - RL epochs (default: 50)
#   BATCH_SIZE       - Batch size (default: 4)
#   SUP_LR           - Supervised learning rate (default: 1e-3)
#   RL_LR            - RL learning rate (default: 3e-4)
#   SEED             - Random seed (default: 42)
#   EXP_NAME         - Experiment name (default: kinematic_pipeline)
#   SKIP_PREPROCESS  - Skip preprocessing if already done (default: 0)

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

DATASET="${1:-full-3d}"
GPU="${GPU:-0}"
SUP_EPOCHS="${SUP_EPOCHS:-100}"
RL_EPOCHS="${RL_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SUP_LR="${SUP_LR:-1e-3}"
RL_LR="${RL_LR:-3e-4}"
SEED="${SEED:-42}"
EXP_NAME="${EXP_NAME:-kinematic_pipeline}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

# Model architecture
NODE_DIM="${NODE_DIM:-128}"
EDGE_DIM="${EDGE_DIM:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
N_LAYERS="${N_LAYERS:-4}"
N_VECTOR_FEATURES="${N_VECTOR_FEATURES:-8}"
MAX_REFINEMENT_STEPS="${MAX_REFINEMENT_STEPS:-4}"
LEVER_DAMPING="${LEVER_DAMPING:-0.95}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="pipeline_${DATASET}_${TIMESTAMP}.log"

echo "========================================================================"
echo "  GraphaRNA - Full Kinematic Refinement Pipeline"
echo "========================================================================"
echo "Dataset:              $DATASET"
echo "Supervised Epochs:    $SUP_EPOCHS (LR: $SUP_LR)"
echo "RL Epochs:            $RL_EPOCHS (LR: $RL_LR)"
echo "Batch Size:           $BATCH_SIZE"
echo "GPU:                  $GPU"
echo "Seed:                 $SEED"
echo "Experiment Name:      $EXP_NAME"
echo "Log File:             $LOG_FILE"
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
echo "Pipeline Steps:"
echo "  [1] Preprocess dataset         (Skip: $SKIP_PREPROCESS)"
echo "  [2] Supervised pre-training    ($SUP_EPOCHS epochs)"
echo "  [3] RL fine-tuning             ($RL_EPOCHS epochs)"
echo "========================================================================"
echo ""

# Redirect all output to log file (but also display on screen)
exec > >(tee -a "$LOG_FILE")
exec 2>&1

# ============================================================================
# Step 1: Preprocessing
# ============================================================================

if [ "$SKIP_PREPROCESS" = "0" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  STEP 1/3: Preprocessing Dataset"
    echo "════════════════════════════════════════════════════════════════════"
    echo "Augmenting PyG pickles with kinematic frame data..."
    echo ""
    
    bash run_preprocess.sh "$DATASET"
    
    echo ""
    echo "✓ Preprocessing complete!"
    echo ""
else
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  STEP 1/3: Preprocessing (SKIPPED)"
    echo "════════════════════════════════════════════════════════════════════"
    echo "Using existing preprocessed data in data/$DATASET/"
    echo ""
fi

# ============================================================================
# Step 2: Supervised Pre-training
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  STEP 2/3: Supervised Pre-training"
echo "════════════════════════════════════════════════════════════════════"
echo "Training KinematicGNN + LDDTSuggester with curriculum noise..."
echo "Curriculum: σ starts at 0.05 rad, increases when BP recovery > 80%"
echo ""

export GPU SEED WANDB
export NODE_DIM EDGE_DIM HIDDEN_DIM N_LAYERS N_VECTOR_FEATURES
export MAX_REFINEMENT_STEPS LEVER_DAMPING

EXP_NAME="${EXP_NAME}_supervised" bash run_supervised.sh \
    "$DATASET" "$SUP_EPOCHS" "$SUP_LR" "$BATCH_SIZE"

SUP_CHECKPOINT="save/${EXP_NAME}_supervised/supervised_epoch_${SUP_EPOCHS}.pt"

echo ""
echo "✓ Supervised pre-training complete!"
echo "  Checkpoint: $SUP_CHECKPOINT"
echo ""

# ============================================================================
# Step 3: RL Fine-tuning
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  STEP 3/3: RL Fine-tuning (PPO)"
echo "════════════════════════════════════════════════════════════════════"
echo "Fine-tuning with physics-based reward (BP + clash + torsion)..."
echo "Curriculum continues: σ increases adaptively until max difficulty"
echo ""

EXP_NAME="${EXP_NAME}_rl" bash run_rl.sh \
    "$SUP_CHECKPOINT" "$DATASET" "$RL_EPOCHS" "$RL_LR" "$BATCH_SIZE"

RL_CHECKPOINT="save/${EXP_NAME}_rl/rl_epoch_${RL_EPOCHS}.pt"

echo ""
echo "✓ RL fine-tuning complete!"
echo "  Checkpoint: $RL_CHECKPOINT"
echo ""

# ============================================================================
# Pipeline Complete
# ============================================================================

echo ""
echo "========================================================================"
echo "  🎉 FULL PIPELINE COMPLETE!"
echo "========================================================================"
echo "Experiment:        $EXP_NAME"
echo "Final checkpoint:  $RL_CHECKPOINT"
echo "Log file:          $LOG_FILE"
echo "------------------------------------------------------------------------"
echo "Checkpoints:"
echo "  Supervised:      $SUP_CHECKPOINT"
echo "  RL:              $RL_CHECKPOINT"
echo "------------------------------------------------------------------------"
echo "Next steps:"
echo "  • Sample structures:"
echo "      bash run_sample.sh $RL_CHECKPOINT input.dotseq"
echo ""
echo "  • Evaluate on test set:"
echo "      python -m grapharna.kinematic.evaluate \\"
echo "          --checkpoint $RL_CHECKPOINT \\"
echo "          --dataset $DATASET \\"
echo "          --split test"
echo ""
echo "  • Continue RL training:"
echo "      bash run_rl.sh $RL_CHECKPOINT $DATASET 50"
echo ""
echo "========================================================================"
