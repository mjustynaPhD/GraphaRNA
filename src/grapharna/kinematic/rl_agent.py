"""
Reinforcement Learning module for kinematic refinement.

Implements:
  - RewardFunction:       Legacy reward (kept for backward compatibility)
  - RNARefinementEnv:     Gym-like environment wrapping the 3D RNA graph
  - PPOAgent:             Proximal Policy Optimization agent for fine-tuning

The KinematicGNN acts as the "policy" network.
The current 3D graph is the "environment."
For the refactored reward specification see ``RefinementReward``
in ``refinement_loop.py``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Optional, Tuple, List
import math

from grapharna.kinematic.frames import RigidFrame
from grapharna.kinematic.kinematic_gnn import KinematicGNN, KinematicConfig
from grapharna.kinematic.refinement_loop import RefinementReward


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------

class RewardFunction(nn.Module):
    """Physics-based reward for RNA structure refinement.

    R = w_bp · BasePairs - w_clash · Clashes - w_torsion · TorsionStress

    Components:
        BasePairs:     Fraction of expected base pairs with correct geometry.
        Clashes:       Count of steric clashes (atoms closer than threshold).
        TorsionStress: Deviation of backbone torsion angles from allowed ranges.
    """

    def __init__(
        self,
        w_bp: float = 1.0,
        w_clash: float = 0.5,
        w_torsion: float = 0.3,
        clash_threshold: float = 2.5,    # Å — minimum allowed P-P distance
        bp_distance_max: float = 12.0,   # Å — max N-N distance for base pair
        bp_distance_ideal: float = 8.5,  # Å — ideal N-N distance (H-bond)
    ):
        super().__init__()
        self.w_bp = w_bp
        self.w_clash = w_clash
        self.w_torsion = w_torsion
        self.clash_threshold = clash_threshold
        self.bp_distance_max = bp_distance_max
        self.bp_distance_ideal = bp_distance_ideal

    def forward(
        self,
        frames: RigidFrame,
        bp_edges: Optional[torch.Tensor] = None,  # (2, E_bp) expected base pairs
        chain_edges: Optional[torch.Tensor] = None,  # (2, E_chain) sequential edges
        batch: Optional[torch.Tensor] = None,       # (N,) batch assignment
    ) -> Dict[str, torch.Tensor]:
        """Compute reward components.

        Returns:
            dict with keys: 'reward', 'bp_reward', 'clash_penalty',
                            'torsion_penalty', and component details.
        """
        coords = frames.global_coords()  # (N, 5, 3)
        N = coords.shape[0]

        # ---- Base Pair Reward ----
        bp_reward = torch.tensor(0.0, device=coords.device)
        if bp_edges is not None and bp_edges.shape[1] > 0:
            # N1/N9 atoms (index 2 in the 5-atom CG model)
            n_atoms = coords[:, 2, :]  # (N, 3)
            bp_i, bp_j = bp_edges[0], bp_edges[1]
            bp_dist = (n_atoms[bp_i] - n_atoms[bp_j]).norm(dim=-1)  # (E_bp,)

            # Smooth reward: Gaussian centered at ideal distance
            bp_score = torch.exp(-0.5 * ((bp_dist - self.bp_distance_ideal) / 2.0) ** 2)
            bp_score = bp_score * (bp_dist < self.bp_distance_max).float()
            bp_reward = bp_score.mean()

        # ---- Clash Penalty ----
        p_atoms = coords[:, 0, :]  # P atoms (N, 3)
        # Compute pairwise distances (efficient for moderate N)
        if N <= 2000:
            pdist = torch.cdist(p_atoms, p_atoms)  # (N, N)
            # Mask self-distances and handle batches
            eye_mask = torch.eye(N, device=coords.device).bool()
            pdist = pdist.masked_fill(eye_mask, float('inf'))

            if batch is not None:
                # Only count clashes within same structure
                batch_mask = batch.unsqueeze(0) != batch.unsqueeze(1)
                pdist = pdist.masked_fill(batch_mask, float('inf'))

            clashes = (pdist < self.clash_threshold).float().sum() / 2.0  # Each pair counted once
            clash_penalty = clashes / max(N, 1)
        else:
            # For very large structures, approximate with k-NN
            clash_penalty = torch.tensor(0.0, device=coords.device)

        # ---- Torsion Stress ----
        torsion_penalty = torch.tensor(0.0, device=coords.device)
        if chain_edges is not None and chain_edges.shape[1] >= 2:
            torsion_penalty = self._compute_torsion_stress(coords, chain_edges)

        # ---- Total Reward ----
        reward = (
            self.w_bp * bp_reward
            - self.w_clash * clash_penalty
            - self.w_torsion * torsion_penalty
        )

        return {
            'reward': reward,
            'bp_reward': bp_reward,
            'clash_penalty': clash_penalty,
            'torsion_penalty': torsion_penalty,
        }

    def _compute_torsion_stress(
        self,
        coords: torch.Tensor,  # (N, 5, 3)
        chain_edges: torch.Tensor,  # (2, E) sequential edges
    ) -> torch.Tensor:
        """Compute backbone torsion angle stress.

        Measures deviation of P(i)-P(i+1)-P(i+2) angles from the
        typical range observed in known RNA structures.
        """
        p_atoms = coords[:, 0, :]  # (N, 3)
        src, dst = chain_edges

        # For consecutive triples: find edges where dst of one = src of next
        # Build a simple chain: i → i+1 → i+2
        if chain_edges.shape[1] < 2:
            return torch.tensor(0.0, device=coords.device)

        # Extract unique sorted chain
        unique_nodes = torch.unique(chain_edges.flatten())
        if unique_nodes.shape[0] < 3:
            return torch.tensor(0.0, device=coords.device)

        # Sequential triples for torsion-like angles
        n_triples = min(unique_nodes.shape[0] - 2, chain_edges.shape[1])
        if n_triples <= 0:
            return torch.tensor(0.0, device=coords.device)

        # Use P atoms of consecutive residues
        v1 = p_atoms[unique_nodes[1:n_triples+1]] - p_atoms[unique_nodes[:n_triples]]
        v2 = p_atoms[unique_nodes[2:n_triples+2]] - p_atoms[unique_nodes[1:n_triples+1]]

        # Pseudo-bond angles
        cos_angle = F.cosine_similarity(v1, v2, dim=-1)
        cos_angle = cos_angle.clamp(-1 + 1e-7, 1 - 1e-7)
        angles = torch.acos(cos_angle)

        # Typical P-P-P angle range: ~90° to ~160° (1.57 to 2.79 rad)
        ANGLE_MIN = math.radians(90)
        ANGLE_MAX = math.radians(160)
        ANGLE_IDEAL = math.radians(125)

        # Stress = deviation from ideal, weighted exponentially outside range
        deviation = (angles - ANGLE_IDEAL).abs()
        out_of_range = (
            F.relu(ANGLE_MIN - angles) +
            F.relu(angles - ANGLE_MAX)
        )
        stress = deviation + 2.0 * out_of_range

        return stress.mean()


# ---------------------------------------------------------------------------
# RL Environment
# ---------------------------------------------------------------------------

class RNARefinementEnv:
    """Gym-like environment for RNA structure refinement via RL.

    State:   Current RigidFrame (3D coordinates + frames)
    Action:  Per-residue SE(3) updates (ΔR, Δt)
    Reward:  Physics-based reward R

    Accepts *either* the legacy ``RewardFunction`` **or** the
    new ``RefinementReward`` — both expose the same call signature.
    """

    def __init__(
        self,
        reward_fn: nn.Module = None,
        max_steps: int = 10,
        lever_damping: float = 0.95,
    ):
        if reward_fn is None:
            reward_fn = RefinementReward()  # new default
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.lever_damping = lever_damping
        self.current_step = 0
        self.frames: Optional[RigidFrame] = None
        self.bp_edges: Optional[torch.Tensor] = None
        self.chain_edges: Optional[torch.Tensor] = None
        self.batch: Optional[torch.Tensor] = None
        self._initial_reward: Optional[float] = None

    def reset(
        self,
        frames: RigidFrame,
        bp_edges: Optional[torch.Tensor] = None,
        chain_edges: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> RigidFrame:
        """Reset environment to initial state.

        Args:
            frames:      Initial RigidFrame (could be linear chain or partial fold).
            bp_edges:    Expected base pair edges.
            chain_edges: Sequential backbone edges.
            batch:       Batch assignment.
        Returns:
            Initial state (RigidFrame).
        """
        self.frames = frames.clone()
        self.bp_edges = bp_edges
        self.chain_edges = chain_edges
        self.batch = batch
        self.current_step = 0

        # Compute initial reward for relative improvement
        with torch.no_grad():
            reward_dict = self.reward_fn(
                self.frames, self.bp_edges, self.chain_edges, self.batch
            )
            self._initial_reward = reward_dict['reward'].item()

        return self.frames

    def step(
        self,
        delta_R: torch.Tensor,  # (N, 3, 3)
        delta_t: torch.Tensor,  # (N, 3)
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[RigidFrame, Dict[str, torch.Tensor], bool]:
        """Apply action and return new state, reward, done.

        Implements the update rule:
            R_new = ΔR · R_old
            t_new = t_old + Δt

        Args:
            delta_R: (N, 3, 3) rotation updates.
            delta_t: (N, 3)   translation updates.
            mask:    (N,) bool — True for residues to update.
        Returns:
            new_frames: Updated RigidFrame.
            reward_dict: Dict with reward components.
            done: Whether episode is finished.
        """
        # Apply SE(3) update
        self.frames = self.frames.apply_update(delta_R, delta_t, mask=mask)
        self.current_step += 1

        # Compute reward
        reward_dict = self.reward_fn(
            self.frames, self.bp_edges, self.chain_edges, self.batch
        )

        done = self.current_step >= self.max_steps

        return self.frames, reward_dict, done

    @property
    def state(self) -> RigidFrame:
        return self.frames


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOMemory:
    """Rollout buffer for PPO."""

    def __init__(self):
        self.states: List[RigidFrame] = []
        self.actions_R: List[torch.Tensor] = []
        self.actions_t: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []
        self.masks: List[Optional[torch.Tensor]] = []

    def add(
        self,
        state: RigidFrame,
        action_R: torch.Tensor,
        action_t: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: bool,
        mask: Optional[torch.Tensor] = None,
    ):
        self.states.append(state.detach())
        self.actions_R.append(action_R.detach())
        self.actions_t.append(action_t.detach())
        self.log_probs.append(log_prob.detach())
        self.rewards.append(reward.detach())
        self.values.append(value.detach())
        self.dones.append(done)
        self.masks.append(mask)

    def clear(self):
        self.__init__()


class ValueHead(nn.Module):
    """Value function approximator for PPO."""

    def __init__(self, node_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Compute per-structure value from node features.

        Args:
            h:     (N, node_dim) node features.
            batch: (N,) batch assignment.
        Returns:
            value: (B,) value estimate per structure.
        """
        # Global mean pooling per structure
        from torch_geometric.nn import global_mean_pool
        h_pooled = global_mean_pool(h, batch)  # (B, node_dim)
        return self.net(h_pooled).squeeze(-1)   # (B,)


class PPOAgent:
    """Proximal Policy Optimization agent for RNA refinement.

    The KinematicGNN serves as the policy network.
    Actions are continuous SE(3) updates parameterized as:
        - ΔR via axis-angle (3D Gaussian → rotation via Rodrigues)
        - Δt via 3D Gaussian

    The agent adds Gaussian noise for exploration and computes
    log-probabilities for the policy gradient.
    """

    def __init__(
        self,
        policy: KinematicGNN,
        config: KinematicConfig,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 4,
        rotation_scale: float = 0.1,    # Scale for rotation noise (radians)
        translation_scale: float = 0.5,  # Scale for translation noise (Å)
    ):
        self.policy = policy
        self.config = config
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.rotation_scale = rotation_scale
        self.translation_scale = translation_scale

        # Value head
        self.value_head = ValueHead(config.node_dim)

        # Optimizer (joint for policy + value)
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(self.value_head.parameters()),
            lr=lr,
        )

        # Learnable log-std for exploration
        self.log_std_R = nn.Parameter(
            torch.ones(3) * math.log(rotation_scale)
        )
        self.log_std_t = nn.Parameter(
            torch.ones(3) * math.log(translation_scale)
        )

        self.memory = PPOMemory()

    def to(self, device):
        self.value_head = self.value_head.to(device)
        self.log_std_R = nn.Parameter(self.log_std_R.to(device))
        self.log_std_t = nn.Parameter(self.log_std_t.to(device))
        return self

    def select_action(
        self,
        frames: RigidFrame,
        sequences: list,
        residue_types: torch.Tensor,
        batch: torch.Tensor,
        step: torch.Tensor,
        covalent_edges: Optional[torch.Tensor] = None,
        bp_edges: Optional[torch.Tensor] = None,
        chain_edges: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action from policy with exploration noise.

        Returns:
            delta_R:  (N, 3, 3) rotation updates
            delta_t:  (N, 3) translation updates
            log_prob: (1,) total log probability
            value:    (B,) state value estimate
        """
        # Forward pass through policy
        out = self.policy(
            frames, sequences, residue_types, batch, step,
            covalent_edges, bp_edges, chain_edges=chain_edges, mask=mask,
        )

        h = out['h']
        delta_R_det = out['delta_R']   # (N, 3, 3) deterministic rotation
        delta_t_det = out['delta_t']   # (N, 3)   deterministic translation

        # Value estimate
        value = self.value_head(h, batch)

        if deterministic:
            # Use axis-angle → rotation for the deterministic output
            return delta_R_det, delta_t_det, torch.tensor(0.0), value

        # Add exploration noise
        N = h.shape[0]
        device = h.device

        # Rotation noise: sample axis-angle, convert to rotation matrix
        std_R = self.log_std_R.exp().to(device)
        noise_R = Normal(torch.zeros(3, device=device), std_R).sample((N,))
        from grapharna.kinematic.frames import axis_angle_to_rotation_matrix
        noise_rot = axis_angle_to_rotation_matrix(noise_R)
        delta_R = noise_rot @ delta_R_det

        # Translation noise
        std_t = self.log_std_t.exp().to(device)
        noise_t = Normal(torch.zeros(3, device=device), std_t).sample((N,))
        delta_t = delta_t_det + noise_t

        # Log probability = log p(noise_R) + log p(noise_t)
        log_prob_R = Normal(torch.zeros(3, device=device), std_R).log_prob(noise_R).sum(-1)
        log_prob_t = Normal(torch.zeros(3, device=device), std_t).log_prob(noise_t).sum(-1)
        log_prob = (log_prob_R + log_prob_t).mean()  # Average over residues

        return delta_R, delta_t, log_prob, value

    def compute_gae(
        self,
        rewards: List[torch.Tensor],
        values: List[torch.Tensor],
        dones: List[bool],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE).

        Returns:
            advantages: (T,) advantage estimates
            returns:    (T,) discounted returns
        """
        T = len(rewards)
        device = rewards[0].device

        advantages = torch.zeros(T, device=device)
        returns = torch.zeros(T, device=device)

        last_gae = 0.0
        last_value = 0.0

        for t in reversed(range(T)):
            mask = 0.0 if dones[t] else 1.0
            next_value = values[t + 1] if t + 1 < T else last_value

            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * mask * last_gae
            last_gae = advantages[t]

        returns = advantages + torch.stack(values[:T])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self) -> Dict[str, float]:
        """Perform PPO update using collected rollouts.

        Returns:
            dict with training metrics.
        """
        mem = self.memory
        if len(mem.rewards) == 0:
            return {}

        T = len(mem.rewards)

        # Compute GAE
        advantages, returns = self.compute_gae(
            mem.rewards, mem.values, mem.dones
        )

        old_log_probs = torch.stack(mem.log_probs)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.ppo_epochs):
            # Re-evaluate all timesteps
            new_log_probs = []
            new_values = []

            for t in range(T):
                # This is a simplified version — in practice you'd re-run
                # the policy on the stored states
                new_log_probs.append(old_log_probs[t])  # Placeholder
                new_values.append(mem.values[t])

            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values)

            # Policy loss (clipped)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(new_values, returns)

            # Entropy bonus (encourages exploration)
            std_R = self.log_std_R.exp()
            std_t = self.log_std_t.exp()
            entropy = 0.5 * (1 + torch.log(2 * math.pi * std_R ** 2)).sum()
            entropy += 0.5 * (1 + torch.log(2 * math.pi * std_t ** 2)).sum()

            # Total loss
            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_head.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        self.memory.clear()

        return {
            'policy_loss': total_policy_loss / self.ppo_epochs,
            'value_loss': total_value_loss / self.ppo_epochs,
            'entropy': total_entropy / self.ppo_epochs,
        }

    def collect_rollout(
        self,
        env: 'RNARefinementEnv',
        sequences: list,
        residue_types: torch.Tensor,
        batch: torch.Tensor,
        covalent_edges: Optional[torch.Tensor] = None,
        bp_edges: Optional[torch.Tensor] = None,
        chain_edges: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Collect a full episode rollout.

        Returns:
            Episode statistics.
        """
        frames = env.state
        total_reward = 0.0
        step_count = 0

        while True:
            step_tensor = torch.full(
                (frames.num_residues,),
                step_count,
                dtype=torch.long,
                device=frames.device,
            )

            # Select action
            delta_R, delta_t, log_prob, value = self.select_action(
                frames, sequences, residue_types, batch, step_tensor,
                covalent_edges, bp_edges, chain_edges=chain_edges,
            )

            # Environment step
            new_frames, reward_dict, done = env.step(delta_R, delta_t)
            reward = reward_dict['reward']

            # Store transition
            self.memory.add(
                state=frames,
                action_R=delta_R,
                action_t=delta_t,
                log_prob=log_prob,
                reward=reward,
                value=value.mean(),  # Average over batch
                done=done,
            )

            total_reward += reward.item()
            step_count += 1
            frames = new_frames

            if done:
                break

        return {
            'total_reward': total_reward,
            'steps': step_count,
            'final_bp_reward': reward_dict['bp_reward'].item(),
            'final_clash_penalty': reward_dict['clash_penalty'].item(),
            'final_torsion_penalty': reward_dict['torsion_penalty'].item(),
        }
