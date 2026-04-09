"""Curriculum Noise Scheduler for Kinematic Refinement Training.

Implements an adaptive noise schedule that starts with ground-truth
structures corrupted by low SO(3) noise (σ = 0.05) and progressively
increases the difficulty when the model proves competent (BP recovery > 80%).

This forces the GNN to learn *local repair* first (correcting small
perturbations) before tackling *global folding* from scratch.

Classes:
    CurriculumNoiseScheduler – manages SO(3) noise σ per training step.
    TrainingScheduler        – wraps the noise scheduler with a rolling-
                               average RL-reward tracker and BP-recovery
                               gating for automatic difficulty escalation.

Usage (standalone):
    >>> scheduler = CurriculumNoiseScheduler(sigma_init=0.05, sigma_max=1.5)
    >>> sigma = scheduler.current_sigma          # 0.05
    >>> frames_noisy = scheduler.apply_noise(frames_true)
    >>> scheduler.step(bp_recovery=0.85)         # triggers σ increase

Usage (with TrainingScheduler wrapping RL):
    >>> ts = TrainingScheduler(sigma_init=0.05, reward_window=50)
    >>> ts.log_step(reward=2.3, bp_recovery=0.82)
    >>> sigma = ts.current_sigma
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from grapharna.kinematic.frames import (
    RigidFrame,
    axis_angle_to_rotation_matrix,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CurriculumNoiseScheduler
# ---------------------------------------------------------------------------

@dataclass
class CurriculumNoiseScheduler:
    """Manages the SO(3) noise level σ applied to ground-truth frames.

    The scheduler starts at a low σ (default 0.05 rad ≈ 3°) so the GNN
    only needs to learn small local corrections.  When the model's
    base-pair (BP) recovery rate exceeds ``bp_threshold`` the σ is
    multiplied by ``sigma_growth_factor``, making the task harder.

    The σ is clamped to ``[sigma_min, sigma_max]`` and separate scaling
    controls how much *translation* noise accompanies the rotation noise.

    Attributes:
        sigma_init:           Initial SO(3) noise standard deviation (radians).
        sigma_min:            Floor for σ (never goes below).
        sigma_max:            Ceiling for σ (caps difficulty).
        sigma_growth_factor:  Multiplicative factor when promotion triggers.
        bp_threshold:         BP recovery rate that triggers promotion.
        translation_ratio:    Translation noise σ_t = σ_R × translation_ratio (Å).
        warmup_steps:         Minimum steps at each σ before promotion.
        current_sigma:        Current SO(3) σ (mutable).
        _steps_at_level:      Steps spent at the current σ level.
        _promotion_count:     How many times σ has been increased.
    """

    sigma_init: float = 0.05
    sigma_min: float = 0.01
    sigma_max: float = 1.5
    sigma_growth_factor: float = 1.3
    bp_threshold: float = 0.80
    translation_ratio: float = 3.0
    warmup_steps: int = 20
    current_sigma: float = field(init=False)
    _steps_at_level: int = field(init=False, default=0)
    _promotion_count: int = field(init=False, default=0)

    def __post_init__(self):
        self.current_sigma = self.sigma_init

    # -- core API ----------------------------------------------------------

    def step(self, bp_recovery: float) -> bool:
        """Update the scheduler after one training epoch / batch.

        Args:
            bp_recovery: Fraction of native base-pairs recovered (0-1).

        Returns:
            True if σ was promoted (increased) this step.
        """
        self._steps_at_level += 1
        promoted = False

        if (
            bp_recovery >= self.bp_threshold
            and self._steps_at_level >= self.warmup_steps
        ):
            new_sigma = self.current_sigma * self.sigma_growth_factor
            new_sigma = min(new_sigma, self.sigma_max)

            if new_sigma > self.current_sigma:
                logger.info(
                    f"[Curriculum] BP recovery {bp_recovery:.2%} >= "
                    f"{self.bp_threshold:.0%} → σ: {self.current_sigma:.4f} "
                    f"→ {new_sigma:.4f} (promotion #{self._promotion_count + 1})"
                )
                self.current_sigma = new_sigma
                self._steps_at_level = 0
                self._promotion_count += 1
                promoted = True

        return promoted

    def reset(self):
        """Reset scheduler to initial state."""
        self.current_sigma = self.sigma_init
        self._steps_at_level = 0
        self._promotion_count = 0

    # -- noise application -------------------------------------------------

    def apply_noise(
        self,
        frames: RigidFrame,
        sigma_override: Optional[float] = None,
    ) -> RigidFrame:
        """Inject isotropic Gaussian SO(3) + translation noise into frames.

        The rotation noise is drawn from an isotropic Gaussian on
        SO(3) via the axis-angle map:
            v ~ N(0, σ² I₃),  R_noise = Rodrigues(v),
            R_noisy = R_noise @ R_true.

        Translation noise:
            t_noisy = t_true + N(0, σ_t² I₃),
            where σ_t = σ × translation_ratio.

        Args:
            frames:         Ground-truth RigidFrame.
            sigma_override: If given, use this σ instead of ``current_sigma``.

        Returns:
            Noisy copy of the input frames.
        """
        sigma = sigma_override if sigma_override is not None else self.current_sigma
        return sample_noisy_frames(
            frames, sigma_R=sigma, sigma_t=sigma * self.translation_ratio
        )

    # -- serialisation -----------------------------------------------------

    def state_dict(self) -> dict:
        """Serialise scheduler state for checkpointing."""
        return {
            'sigma_init': self.sigma_init,
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max,
            'sigma_growth_factor': self.sigma_growth_factor,
            'bp_threshold': self.bp_threshold,
            'translation_ratio': self.translation_ratio,
            'warmup_steps': self.warmup_steps,
            'current_sigma': self.current_sigma,
            '_steps_at_level': self._steps_at_level,
            '_promotion_count': self._promotion_count,
        }

    def load_state_dict(self, state: dict):
        """Restore scheduler state from a checkpoint."""
        for key, val in state.items():
            setattr(self, key, val)

    def __repr__(self) -> str:
        return (
            f"CurriculumNoiseScheduler(σ={self.current_sigma:.4f}, "
            f"promotions={self._promotion_count}, "
            f"steps_at_level={self._steps_at_level})"
        )


# ---------------------------------------------------------------------------
# TrainingScheduler (wraps CurriculumNoiseScheduler + reward tracking)
# ---------------------------------------------------------------------------

class TrainingScheduler:
    """High-level training scheduler managing noise curriculum via RL reward.

    Maintains a rolling window of recent RL rewards and BP recovery rates.
    After each logged step the scheduler checks:
        1. Is the rolling-average BP recovery above ``bp_threshold``?
        2. Have we spent enough warm-up steps at this noise level?
    If both are true, σ is increased automatically.

    The scheduler also provides convenience helpers for:
    - Detecting phase transitions (supervised → RL)
    - Generating noise σ for the current step
    - Logging curriculum statistics to a dict (or wandb)

    Example::

        ts = TrainingScheduler(sigma_init=0.05, reward_window=50)
        for epoch in range(n_epochs):
            for batch in loader:
                ...  # training step
                ts.log_step(reward=r, bp_recovery=bp)
            sigma = ts.current_sigma
            noisy_frames = ts.apply_noise(frames_true)
    """

    def __init__(
        self,
        sigma_init: float = 0.05,
        sigma_min: float = 0.01,
        sigma_max: float = 1.5,
        sigma_growth_factor: float = 1.3,
        bp_threshold: float = 0.80,
        translation_ratio: float = 3.0,
        warmup_steps: int = 20,
        reward_window: int = 50,
    ):
        self.noise_scheduler = CurriculumNoiseScheduler(
            sigma_init=sigma_init,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_growth_factor=sigma_growth_factor,
            bp_threshold=bp_threshold,
            translation_ratio=translation_ratio,
            warmup_steps=warmup_steps,
        )

        self.reward_window = reward_window
        self._reward_history: deque = deque(maxlen=reward_window)
        self._bp_history: deque = deque(maxlen=reward_window)
        self._sigma_history: List[float] = [sigma_init]
        self._total_steps: int = 0
        self._promotions: List[int] = []  # step numbers at which σ was promoted

    # -- properties --------------------------------------------------------

    @property
    def current_sigma(self) -> float:
        """Current SO(3) noise σ."""
        return self.noise_scheduler.current_sigma

    @property
    def rolling_reward(self) -> float:
        """Rolling average of logged RL rewards."""
        if not self._reward_history:
            return 0.0
        return sum(self._reward_history) / len(self._reward_history)

    @property
    def rolling_bp_recovery(self) -> float:
        """Rolling average of logged BP recovery rates."""
        if not self._bp_history:
            return 0.0
        return sum(self._bp_history) / len(self._bp_history)

    @property
    def difficulty_progress(self) -> float:
        """Fraction of the noise range traversed: 0 = easiest, 1 = hardest."""
        sched = self.noise_scheduler
        if sched.sigma_max <= sched.sigma_init:
            return 1.0
        progress = (
            (math.log(sched.current_sigma) - math.log(sched.sigma_init))
            / (math.log(sched.sigma_max) - math.log(sched.sigma_init))
        )
        return max(0.0, min(1.0, progress))

    # -- core API ----------------------------------------------------------

    def log_step(
        self,
        reward: float = 0.0,
        bp_recovery: float = 0.0,
    ) -> bool:
        """Log a training step and possibly promote σ.

        Args:
            reward:      RL reward for this step (used for tracking).
            bp_recovery: Base-pair recovery rate for this step (0-1).

        Returns:
            True if σ was promoted this step.
        """
        self._reward_history.append(reward)
        self._bp_history.append(bp_recovery)
        self._total_steps += 1

        # Use rolling average BP recovery for the promotion decision
        rolling_bp = self.rolling_bp_recovery
        promoted = self.noise_scheduler.step(rolling_bp)

        if promoted:
            self._promotions.append(self._total_steps)

        self._sigma_history.append(self.current_sigma)
        return promoted

    def apply_noise(
        self,
        frames: RigidFrame,
        sigma_override: Optional[float] = None,
    ) -> RigidFrame:
        """Apply current curriculum noise to ground-truth frames."""
        return self.noise_scheduler.apply_noise(frames, sigma_override)

    def reset(self):
        """Reset all state."""
        self.noise_scheduler.reset()
        self._reward_history.clear()
        self._bp_history.clear()
        self._sigma_history = [self.noise_scheduler.sigma_init]
        self._total_steps = 0
        self._promotions.clear()

    # -- stats / serialisation ---------------------------------------------

    def get_stats(self) -> dict:
        """Return a dict of curriculum statistics (for logging / wandb)."""
        return {
            'curriculum/sigma': self.current_sigma,
            'curriculum/rolling_reward': self.rolling_reward,
            'curriculum/rolling_bp_recovery': self.rolling_bp_recovery,
            'curriculum/difficulty_progress': self.difficulty_progress,
            'curriculum/total_steps': self._total_steps,
            'curriculum/promotions': len(self._promotions),
        }

    def state_dict(self) -> dict:
        """Serialise full state for checkpointing."""
        return {
            'noise_scheduler': self.noise_scheduler.state_dict(),
            'reward_history': list(self._reward_history),
            'bp_history': list(self._bp_history),
            'sigma_history': self._sigma_history,
            'total_steps': self._total_steps,
            'promotions': self._promotions,
            'reward_window': self.reward_window,
        }

    def load_state_dict(self, state: dict):
        """Restore from checkpoint."""
        self.noise_scheduler.load_state_dict(state['noise_scheduler'])
        self._reward_history = deque(state['reward_history'], maxlen=self.reward_window)
        self._bp_history = deque(state['bp_history'], maxlen=self.reward_window)
        self._sigma_history = state['sigma_history']
        self._total_steps = state['total_steps']
        self._promotions = state['promotions']

    def __repr__(self) -> str:
        return (
            f"TrainingScheduler(σ={self.current_sigma:.4f}, "
            f"rolling_bp={self.rolling_bp_recovery:.2%}, "
            f"rolling_r={self.rolling_reward:.3f}, "
            f"steps={self._total_steps}, "
            f"promotions={len(self._promotions)})"
        )


# ---------------------------------------------------------------------------
# Standalone noise utility
# ---------------------------------------------------------------------------

def sample_noisy_frames(
    frames: RigidFrame,
    sigma_R: float,
    sigma_t: float,
) -> RigidFrame:
    """Inject isotropic Gaussian noise on SO(3) + ℝ³ into frames.

    Rotation noise:
        v ~ N(0, σ_R² I₃)  →  R_noise = exp(v×)  →  R_noisy = R_noise @ R_true

    Translation noise:
        t_noisy = t_true + N(0, σ_t² I₃)

    Args:
        frames:  Ground-truth RigidFrame.
        sigma_R: Standard deviation for SO(3) axis-angle noise (radians).
        sigma_t: Standard deviation for translation noise (Å).

    Returns:
        A new RigidFrame with noise applied.
    """
    device = frames.device
    num_res = frames.num_residues

    # --- SO(3) noise via axis-angle ---
    axis_angle_noise = torch.randn(num_res, 3, device=device) * sigma_R
    R_noise = axis_angle_to_rotation_matrix(axis_angle_noise)
    R_noisy = R_noise @ frames.R

    # --- Translation noise ---
    t_noise = torch.randn_like(frames.t) * sigma_t
    t_noisy = frames.t + t_noise

    return RigidFrame(R=R_noisy, t=t_noisy, local_coords=frames.local_coords)
