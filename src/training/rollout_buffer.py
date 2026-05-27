"""
rollout_buffer.py
-----------------
Fixed-capacity rollout buffer for PPO on the probe placement POMDP.

Stores complete on-policy trajectories collected by the agent, then
computes Generalised Advantage Estimation (GAE) at the end of each
rollout before the PPO update.

─────────────────────────────────────────────────────────────────
What is stored per transition
─────────────────────────────────────────────────────────────────
Each call to add() stores one (s_t, a_t, r_t, done_t) transition
along with the policy quantities computed at s_t:

  Observation tensors (from env):
    node_features  (MAX_NODES, NODE_FEATURE_DIM)  float32
    edge_index     (2, MAX_EDGES_PADDED)           int64
    node_mask      (MAX_NODES,)                    bool
    edge_mask      (MAX_EDGES_PADDED,)             bool
    slo_health     (NUM_SLOS,)                     float32
    action_mask    (MAX_ACTION_DIM,)               bool

  Action encoding:
    probeable_idx  (MAX_PROBEABLE,)                int64
        Padded with zeros; valid entries = n_p for this episode.
    n_probeable    ()                              int64
        Number of valid probeable nodes in this episode.

  Policy quantities (from act()):
    actions        ()   int64
    log_probs      ()   float32   log π(a_t | s_t)
    values         ()   float32   V(s_t)

  Environment feedback:
    rewards        ()   float32
    dones          ()   bool      terminated OR truncated

Computed after rollout (by compute_gae()):
    advantages     ()   float32   GAE(λ) advantage estimate
    returns        ()   float32   advantages + values  (PPO target for critic)

─────────────────────────────────────────────────────────────────
GAE (Generalised Advantage Estimation)
─────────────────────────────────────────────────────────────────
For t = T-1 down to 0:

    δ_t     = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
    A_t     = δ_t + γ · λ · (1 - done_t) · A_{t+1}
    R_t     = A_t + V(s_t)

where γ is the discount factor and λ is the GAE smoothing parameter.
done_t = 1 at episode boundaries (terminated or truncated) to correctly
bootstrap across episode boundaries in a multi-episode rollout.

─────────────────────────────────────────────────────────────────
Multi-episode rollouts
─────────────────────────────────────────────────────────────────
The buffer holds ROLLOUT_LEN transitions which may span multiple
episodes.  The trainer calls env.reset() when done=True and
continues filling the same buffer.  GAE handles episode boundaries
correctly via the done mask.

─────────────────────────────────────────────────────────────────
Public API
─────────────────────────────────────────────────────────────────
  RolloutBuffer(rollout_len, device)
    .add(obs_dict, episode_graph, action, log_prob, value,
         reward, done)         — store one transition
    .compute_gae(last_value, gamma, gae_lambda)
                               — compute advantages + returns
    .get_batches(batch_size)   — iterate over mini-batches for PPO update
    .reset()                   — clear buffer for next rollout
    .is_full                   — True when rollout_len transitions stored
    .size                      — number of transitions currently stored
"""

from typing import Generator, Optional, Tuple

import numpy as np
import torch

from src.simulator.env.observation_builder import NODE_FEATURE_DIM
from src.simulator.config.slo_config import NUM_SLOS
from src.models.gnn_encoder import MAX_NODES
from src.models.policy_network import MAX_ACTION_DIM, MAX_PROBEABLE, build_probeable_indices

# Edge index padding size — must match env observation_space
MAX_EDGES_PADDED: int = 200


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Fixed-capacity on-policy rollout buffer for PPO.

    Parameters
    ----------
    rollout_len : int
        T_rollout — number of transitions to collect before each PPO update.
        Typical values: 512, 1024, 2048.
    device : torch.device
        Device for all tensors (cpu or cuda).
    """

    def __init__(self, rollout_len: int, device: torch.device):
        assert rollout_len > 0
        self.rollout_len = rollout_len
        self.device      = device
        self._ptr        = 0          # next write position
        self._full       = False      # True once rollout_len steps stored
        self._gae_done   = False      # True once compute_gae() has been called

        # Pre-allocate all storage tensors
        self._alloc()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of transitions currently stored."""
        return self.rollout_len if self._full else self._ptr

    @property
    def is_full(self) -> bool:
        """True when rollout_len transitions have been stored."""
        return self._full

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def _alloc(self) -> None:
        """Pre-allocate all storage tensors on self.device."""
        T  = self.rollout_len
        d  = self.device

        # Observation tensors
        self.node_features  = torch.zeros(T, MAX_NODES, NODE_FEATURE_DIM,
                                          dtype=torch.float32, device=d)
        self.edge_index     = torch.zeros(T, 2, MAX_EDGES_PADDED,
                                          dtype=torch.int64,   device=d)
        self.node_mask      = torch.zeros(T, MAX_NODES,
                                          dtype=torch.bool,    device=d)
        self.edge_mask      = torch.zeros(T, MAX_EDGES_PADDED,
                                          dtype=torch.bool,    device=d)
        self.slo_health     = torch.zeros(T, NUM_SLOS,
                                          dtype=torch.float32, device=d)
        self.action_mask    = torch.zeros(T, MAX_ACTION_DIM,
                                          dtype=torch.bool,    device=d)

        # Action encoding
        self.probeable_idx  = torch.zeros(T, MAX_PROBEABLE,
                                          dtype=torch.int64,   device=d)
        self.n_probeable    = torch.zeros(T,
                                          dtype=torch.int64,   device=d)

        # Policy quantities
        self.actions        = torch.zeros(T, dtype=torch.int64,   device=d)
        self.log_probs      = torch.zeros(T, dtype=torch.float32, device=d)
        self.values         = torch.zeros(T, dtype=torch.float32, device=d)

        # Environment feedback
        self.rewards        = torch.zeros(T, dtype=torch.float32, device=d)
        self.dones          = torch.zeros(T, dtype=torch.bool,    device=d)

        # Computed by compute_gae()
        self.advantages     = torch.zeros(T, dtype=torch.float32, device=d)
        self.returns        = torch.zeros(T, dtype=torch.float32, device=d)

    # ------------------------------------------------------------------
    # add()
    # ------------------------------------------------------------------

    def add(
        self,
        obs_dict:      dict,
        episode_graph,
        action:        int,
        log_prob:      float,
        value:         float,
        reward:        float,
        done:          bool,
    ) -> None:
        """
        Store one transition into the buffer.

        Must be called with the observation BEFORE the action was applied
        (i.e. s_t, not s_{t+1}).

        Parameters
        ----------
        obs_dict : dict
            Observation from env.reset() or env.step() at time t.
            Keys: node_features, edge_index, slo_health, action_mask.
        episode_graph : EpisodeGraph
            Active episode graph at time t (used to build probeable_idx).
        action : int
            Action taken at time t.
        log_prob : float
            log π(a_t | s_t) from the policy at time t.
        value : float
            V(s_t) from the critic at time t.
        reward : float
            r_t received after taking action a_t.
        done : bool
            True if this transition ends an episode
            (terminated OR truncated).
        """
        assert not self._full, \
            "Buffer is full — call reset() before adding more transitions"

        i = self._ptr

        # ── Observation tensors ───────────────────────────────────────
        self.node_features[i] = torch.tensor(
            obs_dict["node_features"], dtype=torch.float32, device=self.device)
        self.edge_index[i]    = torch.tensor(
            obs_dict["edge_index"],    dtype=torch.int64,   device=self.device)
        self.slo_health[i]    = torch.tensor(
            obs_dict["slo_health"],    dtype=torch.float32, device=self.device)
        self.action_mask[i]   = torch.tensor(
            obs_dict["action_mask"],   dtype=torch.bool,    device=self.device)

        # ── Masks ─────────────────────────────────────────────────────
        from src.models.gnn_encoder import build_masks
        nm, em = build_masks(episode_graph)
        self.node_mask[i] = nm.to(self.device)
        self.edge_mask[i] = em.to(self.device)

        # ── Probeable node indices ────────────────────────────────────
        pi    = build_probeable_indices(episode_graph)
        n_p   = len(pi)
        self.probeable_idx[i, :n_p] = pi.to(self.device)
        self.n_probeable[i]          = n_p

        # ── Policy quantities ─────────────────────────────────────────
        self.actions[i]   = action
        self.log_probs[i] = log_prob
        self.values[i]    = value

        # ── Environment feedback ──────────────────────────────────────
        self.rewards[i] = reward
        self.dones[i]   = done

        # ── Advance pointer ───────────────────────────────────────────
        self._ptr += 1
        if self._ptr == self.rollout_len:
            self._full = True

    # ------------------------------------------------------------------
    # compute_gae()
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        last_value: float,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Compute GAE advantages and returns for the stored rollout.

        Must be called after the rollout is complete (is_full=True or
        at a partial rollout end) and before get_batches().

        Parameters
        ----------
        last_value : float
            V(s_{T}) — critic estimate of the value of the state AFTER
            the last stored transition.  Used to bootstrap the return
            for non-terminal transitions at the rollout boundary.
            Pass 0.0 if the last transition was a terminal state.
        gamma : float
            Discount factor γ.
        gae_lambda : float
            GAE smoothing parameter λ.
            λ=1 → Monte Carlo returns (high variance, low bias).
            λ=0 → TD(0) (low variance, high bias).
            λ=0.95 is the standard PPO default.
        """
        T = self.size

        # Work on CPU numpy for the backward pass loop (faster than GPU)
        rewards  = self.rewards[:T].cpu().numpy()
        values   = self.values[:T].cpu().numpy()
        dones    = self.dones[:T].cpu().numpy().astype(np.float32)

        advantages = np.zeros(T, dtype=np.float32)
        last_gae   = 0.0
        next_value = float(last_value)

        # Backward pass: t = T-1 down to 0
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t]

            # TD residual
            delta = (
                rewards[t]
                + gamma * next_value * next_non_terminal
                - values[t]
            )

            # GAE recurrence
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

            advantages[t] = last_gae
            next_value    = values[t]

        returns = advantages + values

        # Store back as tensors
        self.advantages[:T] = torch.tensor(advantages, dtype=torch.float32,
                                           device=self.device)
        self.returns[:T]    = torch.tensor(returns,    dtype=torch.float32,
                                           device=self.device)

        self._gae_done = True

    # ------------------------------------------------------------------
    # get_batches()
    # ------------------------------------------------------------------

    def get_batches(
        self,
        batch_size: int,
        normalize_advantages: bool = True,
    ) -> Generator[dict, None, None]:
        """
        Iterate over randomly shuffled mini-batches for the PPO update.

        Must be called after compute_gae().

        Parameters
        ----------
        batch_size : int
            Mini-batch size.  If rollout_len is not divisible by batch_size,
            the last mini-batch may be smaller.
        normalize_advantages : bool
            If True, normalise advantages to zero mean and unit std within
            the full rollout before batching.  Standard PPO practice —
            stabilises policy gradient magnitudes.

        Yields
        ------
        batch : dict
            All fields sliced to (batch_size, ...) tensors, ready to be
            passed directly to ppo.update().
            Keys match buffer field names plus 'old_log_probs' (alias
            for log_probs stored at collection time).
        """
        assert self._gae_done, \
            "Call compute_gae() before get_batches()"

        T = self.size

        # Normalise advantages over the full rollout
        adv = self.advantages[:T].clone()
        if normalize_advantages and T > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Random permutation for mini-batch sampling
        indices = torch.randperm(T, device=self.device)

        for start in range(0, T, batch_size):
            idx = indices[start : start + batch_size]

            yield {
                # Observation
                "node_features":  self.node_features[idx],
                "edge_index":     self.edge_index[idx],
                "node_mask":      self.node_mask[idx],
                "edge_mask":      self.edge_mask[idx],
                "slo_health":     self.slo_health[idx],
                "action_mask":    self.action_mask[idx],
                # Action encoding
                "probeable_idx":  self.probeable_idx[idx],
                "n_probeable":    self.n_probeable[idx],
                # Policy quantities at collection time
                "actions":        self.actions[idx],
                "old_log_probs":  self.log_probs[idx],
                "old_values":     self.values[idx],
                # GAE outputs
                "advantages":     adv[idx],
                "returns":        self.returns[idx],
            }

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Clear the buffer for the next rollout.

        Zeros all storage tensors and resets the pointer.
        Call this after each PPO update.
        """
        self._ptr      = 0
        self._full     = False
        self._gae_done = False

        # Zero all tensors (avoids stale data from previous rollout)
        for attr in [
            "node_features", "edge_index", "node_mask", "edge_mask",
            "slo_health", "action_mask", "probeable_idx", "n_probeable",
            "actions", "log_probs", "values", "rewards", "dones",
            "advantages", "returns",
        ]:
            getattr(self, attr).zero_()