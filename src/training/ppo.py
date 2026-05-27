"""
ppo.py
------
Proximal Policy Optimisation (PPO) update for the probe placement POMDP.

Implements the clipped PPO objective (Schulman et al., 2017) with:
  - Clipped surrogate policy loss       L_clip
  - Clipped value function loss         L_value  (optional, recommended)
  - Entropy bonus                       L_entropy
  - Gradient norm clipping

Total loss:
    L = L_clip + c1 · L_value − c2 · L_entropy

─────────────────────────────────────────────────────────────────
PPO update cycle (called by trainer.py)
─────────────────────────────────────────────────────────────────
  1. Collect rollout_len transitions → RolloutBuffer
  2. compute_gae(last_value)
  3. For epoch in range(n_epochs):
       For batch in buffer.get_batches(batch_size):
           forward pass  → new log_probs, values, entropy
           compute losses
           backprop + gradient clip + optimizer step
  4. buffer.reset()

─────────────────────────────────────────────────────────────────
Clipped value loss
─────────────────────────────────────────────────────────────────
Standard PPO clips the value function loss to prevent large updates:

    V_clipped  = V_old + clip(V_new − V_old, −ε, +ε)
    L_value    = 0.5 · mean( max( (V_new − R)², (V_clipped − R)² ) )

This is optional but recommended — it keeps the critic stable when
the policy changes rapidly.

─────────────────────────────────────────────────────────────────
Public API
─────────────────────────────────────────────────────────────────
  PPOConfig             — hyperparameters
  PPOStats              — per-update diagnostic metrics
  PPO(policy, config)   — update engine
    .update(buffer)     → PPOStats
    .optimizer          — the Adam optimizer (for checkpointing)
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from src.models.policy_network import PolicyNetwork, MAX_PROBEABLE


# ---------------------------------------------------------------------------
# PPOConfig
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """
    Hyperparameters for the PPO update.

    Attributes
    ----------
    learning_rate : float
        Adam learning rate.
    n_epochs : int
        Number of PPO epochs per rollout (how many times we iterate over
        the same batch of transitions).  Typical: 4–10.
    batch_size : int
        Mini-batch size within each PPO epoch.
    clip_epsilon : float
        ε — ratio clipping range [1−ε, 1+ε].  Standard: 0.2.
    value_coef : float
        c1 — weight of the value loss term.  Standard: 0.5.
    entropy_coef : float
        c2 — weight of the entropy bonus.  Higher → more exploration.
        Standard: 0.01.  Anneal over training for curriculum.
    max_grad_norm : float
        Gradient norm clipping threshold.  Standard: 0.5.
    clip_value_loss : bool
        If True, apply PPO-style clipping to the value loss as well.
        Recommended: True.
    normalize_advantages : bool
        If True, normalise advantages to zero mean / unit std within
        each rollout before the PPO update.  Standard: True.
    """
    learning_rate:        float = 3e-4
    n_epochs:             int   = 4
    batch_size:           int   = 64
    clip_epsilon:         float = 0.2
    value_coef:           float = 0.5
    entropy_coef:         float = 0.01
    max_grad_norm:        float = 0.5
    clip_value_loss:      bool  = True
    normalize_advantages: bool  = True

    def __post_init__(self):
        assert self.learning_rate  > 0
        assert self.n_epochs       >= 1
        assert self.batch_size     >= 1
        assert self.clip_epsilon   > 0
        assert self.value_coef     >= 0
        assert self.entropy_coef   >= 0
        assert self.max_grad_norm  > 0


# ---------------------------------------------------------------------------
# PPOStats
# ---------------------------------------------------------------------------

@dataclass
class PPOStats:
    """
    Diagnostic metrics averaged over all mini-batch updates in one
    PPO update call (n_epochs × n_batches updates total).

    Attributes
    ----------
    total_loss    : float   L_clip + c1·L_value − c2·L_entropy
    policy_loss   : float   L_clip (clipped surrogate objective)
    value_loss    : float   L_value (critic MSE, possibly clipped)
    entropy       : float   Mean action distribution entropy H(π)
    approx_kl     : float   Approximate KL(π_old || π_new) for monitoring
                            approx_kl = mean(log_prob_old − log_prob_new)
    clip_fraction : float   Fraction of transitions where ratio was clipped
    explained_var : float   1 − Var(R − V) / Var(R) — critic quality
                            1.0 = perfect, 0.0 = as good as mean baseline,
                            negative = worse than baseline
    n_updates     : int     Number of mini-batch gradient steps taken
    """
    total_loss:    float = 0.0
    policy_loss:   float = 0.0
    value_loss:    float = 0.0
    entropy:       float = 0.0
    approx_kl:     float = 0.0
    clip_fraction: float = 0.0
    explained_var: float = 0.0
    n_updates:     int   = 0

    def __truediv__(self, n: int) -> "PPOStats":
        """Average stats over n updates."""
        return PPOStats(
            total_loss    = self.total_loss    / n,
            policy_loss   = self.policy_loss   / n,
            value_loss    = self.value_loss    / n,
            entropy       = self.entropy       / n,
            approx_kl     = self.approx_kl     / n,
            clip_fraction = self.clip_fraction / n,
            explained_var = self.explained_var,   # not averaged (set once)
            n_updates     = n,
        )

    def __add__(self, other: "PPOStats") -> "PPOStats":
        return PPOStats(
            total_loss    = self.total_loss    + other.total_loss,
            policy_loss   = self.policy_loss   + other.policy_loss,
            value_loss    = self.value_loss    + other.value_loss,
            entropy       = self.entropy       + other.entropy,
            approx_kl     = self.approx_kl     + other.approx_kl,
            clip_fraction = self.clip_fraction + other.clip_fraction,
            explained_var = self.explained_var + other.explained_var,
            n_updates     = self.n_updates     + other.n_updates,
        )


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

class PPO:
    """
    PPO update engine.

    Parameters
    ----------
    policy : PolicyNetwork
        The actor-critic network to train.
    config : PPOConfig
        Training hyperparameters.
    device : torch.device
        Device for all tensors.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        config: Optional[PPOConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.policy = policy
        self.cfg    = config or PPOConfig()
        self.device = device or torch.device("cpu")

        self.policy.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.cfg.learning_rate,
            eps=1e-5,   # Adam ε — standard for PPO
        )

    # ------------------------------------------------------------------
    # update()
    # ------------------------------------------------------------------

    def update(self, buffer) -> PPOStats:
        """
        Run n_epochs PPO epochs over the rollout buffer.

        Parameters
        ----------
        buffer : RolloutBuffer
            Filled buffer with compute_gae() already called.

        Returns
        -------
        PPOStats
            Averaged diagnostic metrics over all mini-batch updates.
        """
        self.policy.train()

        # Accumulated stats over all mini-batch updates
        acc = PPOStats()
        n_updates = 0

        # Explained variance — computed once over full rollout
        with torch.no_grad():
            all_returns = buffer.returns[:buffer.size]
            all_values  = buffer.values[:buffer.size]
            explained_var = self._explained_variance(all_returns, all_values)

        for epoch in range(self.cfg.n_epochs):
            for batch in buffer.get_batches(
                self.cfg.batch_size,
                normalize_advantages=self.cfg.normalize_advantages,
            ):
                stats = self._update_batch(batch)
                acc   = acc + stats
                n_updates += 1

        # Average over all updates
        result = acc / n_updates
        result.explained_var = float(explained_var)
        return result

    # ------------------------------------------------------------------
    # _update_batch()  — single mini-batch gradient step
    # ------------------------------------------------------------------

    def _update_batch(self, batch: dict) -> PPOStats:
        """
        Compute PPO losses for one mini-batch and take a gradient step.

        Parameters
        ----------
        batch : dict
            Mini-batch from RolloutBuffer.get_batches().

        Returns
        -------
        PPOStats with raw (non-averaged) values for this batch.
        """
        # ── Unpack batch ──────────────────────────────────────────────
        node_features = batch["node_features"].to(self.device)
        edge_index    = batch["edge_index"].to(self.device)
        node_mask     = batch["node_mask"].to(self.device)
        edge_mask     = batch["edge_mask"].to(self.device)
        slo_health    = batch["slo_health"].to(self.device)
        action_mask   = batch["action_mask"].to(self.device)
        probeable_idx = batch["probeable_idx"].to(self.device)
        actions       = batch["actions"].to(self.device)           # (B,)
        old_log_probs = batch["old_log_probs"].to(self.device)     # (B,)
        old_values    = batch["old_values"].to(self.device)        # (B,)
        advantages    = batch["advantages"].to(self.device)        # (B,) normalised
        returns       = batch["returns"].to(self.device)           # (B,)

        # ── Forward pass ──────────────────────────────────────────────
        out = self.policy(
            node_features, edge_index, node_mask, edge_mask,
            slo_health, action_mask, probeable_idx,
        )

        # New log_probs for the actions that were actually taken
        # out.log_probs: (B, MAX_ACTION_DIM) → gather action column → (B,)
        new_log_probs = out.log_probs.gather(
            dim=1, index=actions.unsqueeze(1)
        ).squeeze(1)                                               # (B,)

        new_values  = out.state_value                              # (B,)
        entropy     = out.entropy                                  # (B,)

        # ── Policy loss  L_clip ───────────────────────────────────────
        # Importance sampling ratio r_t = π_new(a|s) / π_old(a|s)
        log_ratio = new_log_probs - old_log_probs
        ratio     = log_ratio.exp()                                # (B,)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.cfg.clip_epsilon,
            1.0 + self.cfg.clip_epsilon,
        ) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # ── Value loss  L_value ───────────────────────────────────────
        if self.cfg.clip_value_loss:
            # Clip value updates to prevent large deviations from old values
            values_clipped = old_values + torch.clamp(
                new_values - old_values,
                -self.cfg.clip_epsilon,
                +self.cfg.clip_epsilon,
            )
            v_loss_unclipped = (new_values    - returns) ** 2
            v_loss_clipped   = (values_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            value_loss = 0.5 * ((new_values - returns) ** 2).mean()

        # ── Entropy bonus  L_entropy ──────────────────────────────────
        entropy_loss = entropy.mean()

        # ── Total loss ────────────────────────────────────────────────
        total_loss = (
            policy_loss
            + self.cfg.value_coef    * value_loss
            - self.cfg.entropy_coef  * entropy_loss
        )

        # ── Gradient step ─────────────────────────────────────────────
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.cfg.max_grad_norm,
        )
        self.optimizer.step()

        # ── Diagnostics ───────────────────────────────────────────────
        with torch.no_grad():
            # Approximate KL divergence (monitoring only — not used in loss)
            # approx_kl ≈ mean(log_prob_old - log_prob_new)
            approx_kl = ((old_log_probs - new_log_probs)
                         .mean().item())

            # Fraction of transitions where ratio was clipped
            clip_fraction = (
                ((ratio - 1.0).abs() > self.cfg.clip_epsilon)
                .float().mean().item()
            )

        return PPOStats(
            total_loss    = total_loss.item(),
            policy_loss   = policy_loss.item(),
            value_loss    = value_loss.item(),
            entropy       = entropy_loss.item(),
            approx_kl     = approx_kl,
            clip_fraction = clip_fraction,
            n_updates     = 1,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _explained_variance(
        returns: torch.Tensor,
        values:  torch.Tensor,
    ) -> float:
        """
        Explained variance of the value function.

        ev = 1 − Var(R − V) / Var(R)

        Interpretation:
          ev = 1.0  → critic perfectly predicts returns
          ev = 0.0  → critic no better than predicting the mean
          ev < 0.0  → critic is worse than predicting the mean
        """
        var_returns = returns.var()
        if var_returns < 1e-8:
            return float("nan")   # degenerate: all returns identical
        residual_var = (returns - values).var()
        return float(1.0 - residual_var / var_returns)

    def set_learning_rate(self, lr: float) -> None:
        """Update the learning rate (used by curriculum scheduler)."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def set_entropy_coef(self, coef: float) -> None:
        """Update the entropy coefficient (used by curriculum scheduler)."""
        self.cfg.entropy_coef = coef