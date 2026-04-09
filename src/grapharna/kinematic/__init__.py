"""
Recursive SE(3)-Equivariant Frame Refinement modules for GraphaRNA.

Architecture (refactored):
  1. Multimodal Input:   RiNALMo (1280-dim) + 2D structure features
  2. GNN Stage:          VN-EGNN layers on spatial k-NN graph
  3. Transformer Bridge: BiasedTransformerStack with B_{ij} base-pair bias
  4. Output Head:        FrameUpdateHead → ΔR (quat/6D) + Δt per residue
  5. Refinement Loop:    RecursiveRefinementLoop with physics-based reward
"""

from .frames import RigidFrame, initialize_linear_chain, forward_kinematics
from .kinematic_gnn import KinematicGNN, KinematicConfig, SecondaryStructureEncoder
from .equivariant_layers import EGNN_Layer, VN_EGNN_Layer
from .frame_update_head import FrameUpdateHead
from .biased_transformer import (
    BiasedTransformerLayer,
    BiasedTransformerStack,
    BiasedMultiHeadAttention,
    build_bp_bias_matrix,
)
from .refinement_loop import RecursiveRefinementLoop, RefinementReward
from .rl_agent import PPOAgent, RNARefinementEnv, RewardFunction
from .lddt_suggester import LDDTSuggester
from .recursive_refiner import RecursiveRefiner
from .curriculum_scheduler import (
    CurriculumNoiseScheduler,
    TrainingScheduler,
    sample_noisy_frames,
)

__all__ = [
    # Frames
    "RigidFrame",
    "initialize_linear_chain",
    "forward_kinematics",
    # Model backbone
    "KinematicGNN",
    "KinematicConfig",
    "SecondaryStructureEncoder",
    # Equivariant layers (GNN stage)
    "EGNN_Layer",
    "VN_EGNN_Layer",
    # Frame update head (output)
    "FrameUpdateHead",
    # Biased transformer (bridge)
    "BiasedTransformerLayer",
    "BiasedTransformerStack",
    "BiasedMultiHeadAttention",
    "build_bp_bias_matrix",
    # Recursive refinement loop
    "RecursiveRefinementLoop",
    "RefinementReward",
    # RL agent
    "PPOAgent",
    "RNARefinementEnv",
    "RewardFunction",
    # Utilities
    "LDDTSuggester",
    "RecursiveRefiner",
    "CurriculumNoiseScheduler",
    "TrainingScheduler",
    "sample_noisy_frames",
]
