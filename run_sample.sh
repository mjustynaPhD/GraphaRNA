#!/bin/bash
#
# Run structure sampling/inference with a trained model.
#
# Given a checkpoint and input .dotseq file, generate RNA 3D structures.
# The dotseq file format:
#   >name
#   SEQUENCE
#   DOT-BRACKET-NOTATION
#
# Usage:
#   bash run_sample.sh CHECKPOINT INPUT_DOTSEQ
#
# Example:
#   bash run_sample.sh save/kinematic_rl/rl_epoch_50.pt input.dotseq
#   GPU=1 bash run_sample.sh save/exp/rl_epoch_100.pt data/test_sequences.dotseq

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

if [ -z "$1" ]; then
    echo "Error: Checkpoint path required!"
    echo "Usage: bash run_sample.sh CHECKPOINT INPUT_DOTSEQ"
    echo ""
    echo "Example:"
    echo "  bash run_sample.sh save/kinematic_rl/rl_epoch_50.pt input.dotseq"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: Input dotseq file required!"
    echo "Usage: bash run_sample.sh CHECKPOINT INPUT_DOTSEQ"
    echo ""
    echo "Example:"
    echo "  bash run_sample.sh save/kinematic_rl/rl_epoch_50.pt input.dotseq"
    exit 1
fi

CHECKPOINT="$1"
INPUT_DOTSEQ="$2"
GPU="${GPU:-0}"
EXP_NAME="${EXP_NAME:-kinematic_sample}"

# Model architecture (must match checkpoint)
NODE_DIM="${NODE_DIM:-128}"
EDGE_DIM="${EDGE_DIM:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
N_LAYERS="${N_LAYERS:-4}"
N_VECTOR_FEATURES="${N_VECTOR_FEATURES:-8}"
MAX_REFINEMENT_STEPS="${MAX_REFINEMENT_STEPS:-4}"
LEVER_DAMPING="${LEVER_DAMPING:-0.95}"

echo "========================================================================"
echo "  GraphaRNA - Structure Sampling (Kinematic Refinement)"
echo "========================================================================"
echo "Checkpoint:        $CHECKPOINT"
echo "Input:             $INPUT_DOTSEQ"
echo "GPU:               $GPU"
echo "Output dir:        samples/$EXP_NAME/"
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

# Check files exist
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$INPUT_DOTSEQ" ]; then
    echo "Error: Input file not found: $INPUT_DOTSEQ"
    exit 1
fi

# ============================================================================
# Run sampling
# ============================================================================

python -m grapharna.kinematic.main \
    --phase sample \
    --checkpoint "$CHECKPOINT" \
    --input "$INPUT_DOTSEQ" \
    --gpu "$GPU" \
    --exp-name "$EXP_NAME" \
    --node-dim "$NODE_DIM" \
    --edge-dim "$EDGE_DIM" \
    --hidden-dim "$HIDDEN_DIM" \
    --n-layers "$N_LAYERS" \
    --n-vector-features "$N_VECTOR_FEATURES" \
    --max-refinement-steps "$MAX_REFINEMENT_STEPS" \
    --lever-damping "$LEVER_DAMPING"

echo ""
echo "========================================================================"
echo "  Sampling complete!"
echo "========================================================================"
echo "Generated structures saved in: samples/$EXP_NAME/"
echo ""
echo "To convert to PDB format, use:"
echo "  python -m grapharna.utils.frames_to_pdb samples/$EXP_NAME/*.pt"
echo ""
