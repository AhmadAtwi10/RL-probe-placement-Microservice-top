"""
policy_network.py
-----------------
PPO Actor-Critic policy network for the probe placement POMDP.

Sits on top of the GNN encoder and produces:
  - action logits  (masked softmax over the action space)
  - state value    V(s)

─────────────────────────────────────────────────────────────────
Design rationale (from our discussion)
─────────────────────────────────────────────────────────────────
Mean pooling destroys node identity, so the actor NEVER uses
the global graph embedding z directly for per-node actions.

  ACTOR  uses per-node embeddings h_v  — node identity preserved
  CRITIC uses global embedding    z    — only needs overall state quality

Concretely:

  logit_add(v)    = MLP_add( h_v )                    scalar per probeable node
  logit_remove(v) = MLP_rem( h_v )                    scalar per probeable node
  logit_noop      = MLP_noop( concat(z, H_t) )        scalar (global context)

  action_logits   = concat(logit_noop, logits_add, logits_remove)   (2|V_p|+1,)
  action_probs    = softmax( action_logits * action_mask )

  V(s)            = MLP_value( concat(z, H_t) )       scalar

─────────────────────────────────────────────────────────────────
Variable action space handling
─────────────────────────────────────────────────────────────────
|V_p| changes across episodes. We always output MAX_ACTION_DIM=45
logits (matching the env's fixed action_space.n). Logits beyond
the current episode's 2|V_p|+1 are set to -inf before softmax
via the action_mask from the environment.

─────────────────────────────────────────────────────────────────
Probeable node index mapping
─────────────────────────────────────────────────────────────────
node_embeddings has shape (MAX_NODES, H) with rows ordered by
episode_graph.present_nodes. Probeable nodes are a subset of
present nodes. The policy uses a probeable_indices tensor of
shape (|V_p|,) to gather the correct h_v rows for each
probeable node, in the same order as the env's action encoding.

─────────────────────────────────────────────────────────────────
Public API
─────────────────────────────────────────────────────────────────
  PolicyConfig              — network hyperparameters
  PolicyNetwork(gnn_config, policy_config)   — nn.Module
    .forward(node_features, edge_index, node_mask, edge_mask,
             slo_health, action_mask, probeable_indices)
      → PolicyOutput(action_logits, action_probs, log_probs,
                     state_value, entropy)
    .act(obs_dict, episode_graph)
      → action (int), log_prob (float), value (float)

  build_probeable_indices(episode_graph) → LongTensor (|V_p|,)
      Maps probeable node position → present_nodes index.
"""

from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.simulator.config.slo_config import NUM_SLOS
from src.simulator.env.observation_builder import NODE_FEATURE_DIM
from src.models.gnn_encoder import (
    GNNConfig, GNNEncoder, build_masks,
    MAX_NODES, MAX_EDGES,
)

# Maximum action space size — matches env's action_space.n
# = 2 * max_probeable_nodes + 1 = 2 * 22 + 1 = 45
MAX_PROBEABLE:  int = 22
MAX_ACTION_DIM: int = 2 * MAX_PROBEABLE + 1   # 45


# ---------------------------------------------------------------------------
# PolicyConfig
# ---------------------------------------------------------------------------

@dataclass
class PolicyConfig:
    """
    Hyperparameters for the actor-critic policy heads.

    Attributes
    ----------
    actor_hidden_dim : int
        Hidden layer size for the per-node actor MLPs (add/remove heads).
    critic_hidden_dim : int
        Hidden layer size for the critic MLP.
    noop_hidden_dim : int
        Hidden layer size for the no_op scoring MLP.
    dropout : float
        Dropout probability in policy heads.
    """
    actor_hidden_dim:  int   = 64
    critic_hidden_dim: int   = 128
    noop_hidden_dim:   int   = 64
    dropout:           float = 0.1

    def __post_init__(self):
        assert self.actor_hidden_dim  > 0
        assert self.critic_hidden_dim > 0
        assert self.noop_hidden_dim   > 0
        assert 0.0 <= self.dropout < 1.0


# ---------------------------------------------------------------------------
# PolicyOutput
# ---------------------------------------------------------------------------

class PolicyOutput(NamedTuple):
    """
    All outputs from a single policy forward pass.

    Attributes
    ----------
    action_logits : (MAX_ACTION_DIM,) or (B, MAX_ACTION_DIM)
        Raw logits before masking and softmax.
    action_probs  : same shape as action_logits
        Probabilities after masking and softmax.
        Masked actions have probability 0.
    log_probs     : same shape
        Log probabilities (log of action_probs).
        Masked actions have log_prob = -inf.
    state_value   : scalar or (B,)
        V(s) — critic estimate of state value.
    entropy       : scalar or (B,)
        Entropy of the action distribution H(π(·|s)).
        Used as exploration bonus in PPO.
    """
    action_logits: torch.Tensor
    action_probs:  torch.Tensor
    log_probs:     torch.Tensor
    state_value:   torch.Tensor
    entropy:       torch.Tensor


# ---------------------------------------------------------------------------
# Helper: build probeable indices
# ---------------------------------------------------------------------------

def build_probeable_indices(episode_graph) -> torch.LongTensor:
    """
    Build a mapping from probeable action slot → present_nodes row index.

    The env's action encoding uses sorted(ep.probeable_nodes) as the
    ordered list of probeable nodes.  node_embeddings rows follow
    ep.present_nodes order.  This function returns a tensor that maps
    each probeable node's position in the action list to its row in
    node_embeddings.

    Parameters
    ----------
    episode_graph : EpisodeGraph

    Returns
    -------
    probeable_indices : LongTensor  (|V_p|,)
        probeable_indices[i] = row index of probeable_nodes[i] in
        episode_graph.present_nodes.
    """
    probeable_ordered = sorted(episode_graph.probeable_nodes)
    present_nodes     = episode_graph.present_nodes
    indices = [present_nodes.index(node) for node in probeable_ordered]
    return torch.tensor(indices, dtype=torch.long)


# ---------------------------------------------------------------------------
# PolicyNetwork
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    PPO Actor-Critic policy network.

    Combines GNNEncoder with three actor heads and one critic head:
      - MLP_add    : per-node → add_probe logit
      - MLP_remove : per-node → remove_probe logit
      - MLP_noop   : global+H_t → no_op logit
      - MLP_value  : global+H_t → state value V(s)

    Parameters
    ----------
    gnn_config    : GNNConfig
        Architecture of the GNN encoder.
    policy_config : PolicyConfig
        Architecture of the actor-critic heads.
    """

    def __init__(
        self,
        gnn_config:    Optional[GNNConfig]    = None,
        policy_config: Optional[PolicyConfig] = None,
    ):
        super().__init__()

        self.gnn_cfg = gnn_config    or GNNConfig()
        self.pol_cfg = policy_config or PolicyConfig()

        H  = self.gnn_cfg.hidden_dim        # GNN embedding dim
        Ah = self.pol_cfg.actor_hidden_dim  # actor hidden dim
        Ch = self.pol_cfg.critic_hidden_dim # critic hidden dim
        Nh = self.pol_cfg.noop_hidden_dim   # noop hidden dim
        dp = self.pol_cfg.dropout

        # ── GNN encoder ───────────────────────────────────────────────────
        self.gnn = GNNEncoder(self.gnn_cfg)

        # ── Actor: add_probe head  (per-node, uses h_v) ───────────────────
        # Input: h_v  (H,) → scalar logit
        self.actor_add = nn.Sequential(
            nn.Linear(H, Ah),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(Ah, 1),
        )

        # ── Actor: remove_probe head  (per-node, uses h_v) ────────────────
        # Separate weights from add_head: the decision to remove a probe
        # requires different reasoning than the decision to add one
        # (risk-aware vs opportunity-seeking)
        self.actor_remove = nn.Sequential(
            nn.Linear(H, Ah),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(Ah, 1),
        )

        # ── Actor: no_op head  (global context, uses concat(z, H_t)) ──────
        # Input: concat(z, H_t) = (H + NUM_SLOS,) → scalar logit
        self.actor_noop = nn.Sequential(
            nn.Linear(H + NUM_SLOS, Nh),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(Nh, 1),
        )

        # ── Critic: value head  (global context, uses concat(z, H_t)) ─────
        # Input: concat(z, H_t) = (H + NUM_SLOS,) → scalar value
        self.critic = nn.Sequential(
            nn.Linear(H + NUM_SLOS, Ch),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(Ch, Ch // 2),
            nn.ReLU(),
            nn.Linear(Ch // 2, 1),
        )

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        node_features:      torch.Tensor,   # (B, N, D) or (N, D)
        edge_index:         torch.Tensor,   # (B, 2, E) or (2, E)
        node_mask:          torch.Tensor,   # (B, N) or (N,)   bool
        edge_mask:          torch.Tensor,   # (B, E) or (E,)   bool
        slo_health:         torch.Tensor,   # (B, NUM_SLOS) or (NUM_SLOS,)
        action_mask:        torch.Tensor,   # (B, MAX_ACTION_DIM) or (MAX_ACTION_DIM,)  bool
        probeable_indices:  torch.Tensor,   # (B, |V_p|) or (|V_p|,)  long
    ) -> PolicyOutput:
        """
        Forward pass through GNN + actor-critic heads.

        Parameters
        ----------
        node_features     : padded node feature matrix
        edge_index        : padded COO edge index
        node_mask         : True for real nodes
        edge_mask         : True for real edges
        slo_health        : H_t — SLO coverage+health vector
        action_mask       : True for valid actions (from env)
        probeable_indices : maps probeable slot i → row in node_embeddings

        Returns
        -------
        PolicyOutput with action_logits, action_probs, log_probs,
        state_value, entropy.
        """
        unbatched = node_features.dim() == 2
        if unbatched:
            node_features     = node_features.unsqueeze(0)
            edge_index        = edge_index.unsqueeze(0)
            node_mask         = node_mask.unsqueeze(0)
            edge_mask         = edge_mask.unsqueeze(0)
            slo_health        = slo_health.unsqueeze(0)
            action_mask       = action_mask.unsqueeze(0)
            probeable_indices = probeable_indices.unsqueeze(0)

        B   = node_features.size(0)
        n_p = probeable_indices.size(1)   # |V_p| for this episode

        # ── 1. GNN encoder ─────────────────────────────────────────────
        # node_emb: (B, N, H)   graph_emb: (B, H)
        node_emb, graph_emb = self.gnn(
            node_features, edge_index, node_mask, edge_mask
        )

        # ── 2. Global context vector  concat(z, H_t) ───────────────────
        # (B, H + NUM_SLOS)
        global_ctx = torch.cat([graph_emb, slo_health], dim=-1)

        # ── 3. Per-probeable-node embeddings ───────────────────────────
        # Gather h_v for each probeable node using probeable_indices
        # probeable_indices: (B, n_p) → used to index node_emb (B, N, H)
        idx_exp = probeable_indices.unsqueeze(-1).expand(B, n_p, node_emb.size(-1))
        h_prob  = torch.gather(node_emb, dim=1, index=idx_exp)   # (B, n_p, H)

        # ── 4. Actor logits ────────────────────────────────────────────

        # add_probe logits: one per probeable node  (B, n_p)
        logits_add = self.actor_add(h_prob).squeeze(-1)           # (B, n_p)

        # remove_probe logits: one per probeable node  (B, n_p)
        logits_rem = self.actor_remove(h_prob).squeeze(-1)        # (B, n_p)

        # no_op logit: single scalar from global context  (B, 1)
        logit_noop = self.actor_noop(global_ctx)                  # (B, 1)

        # Assemble full logit vector in env action encoding order:
        # [no_op | add_probe(0..n_p-1) | remove_probe(0..n_p-1)]
        # Then pad to MAX_ACTION_DIM with -inf for out-of-range slots
        logits_episode = torch.cat(
            [logit_noop, logits_add, logits_rem], dim=-1
        )   # (B, 2*n_p+1)

        action_logits = torch.full(
            (B, MAX_ACTION_DIM), fill_value=float('-inf'),
            dtype=torch.float32, device=node_features.device,
        )
        action_logits[:, : 2 * n_p + 1] = logits_episode

        # ── 5. Apply action mask ───────────────────────────────────────
        # Invalid actions → -inf so they get probability 0 after softmax
        action_logits = action_logits.masked_fill(~action_mask, float('-inf'))

        # ── 6. Action probabilities and log probs ──────────────────────
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs    = F.log_softmax(action_logits, dim=-1)

        # ── 7. Entropy  H(π) = -Σ p·log(p)  (over valid actions only) ─
        # Replace -inf log_probs with 0 for entropy computation
        # (0·log(0) = 0 by convention)
        log_probs_safe = log_probs.clone()
        log_probs_safe[~action_mask] = 0.0
        entropy = -(action_probs * log_probs_safe).sum(dim=-1)   # (B,)

        # ── 8. State value ─────────────────────────────────────────────
        state_value = self.critic(global_ctx).squeeze(-1)         # (B,)

        # ── Remove batch dim if input was unbatched ────────────────────
        if unbatched:
            action_logits = action_logits.squeeze(0)
            action_probs  = action_probs.squeeze(0)
            log_probs     = log_probs.squeeze(0)
            state_value   = state_value.squeeze(0)
            entropy       = entropy.squeeze(0)

        return PolicyOutput(
            action_logits = action_logits,
            action_probs  = action_probs,
            log_probs     = log_probs,
            state_value   = state_value,
            entropy       = entropy,
        )

    # ------------------------------------------------------------------
    # Convenience method for environment interaction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        obs_dict:       dict,
        episode_graph,
        deterministic:  bool = False,
    ) -> tuple:
        """
        Sample or select an action from the current observation.

        Used during rollout collection (training) and evaluation.

        Parameters
        ----------
        obs_dict : dict
            Observation dict from env.reset() or env.step().
            Keys: node_features, edge_index, slo_health, action_mask.
        episode_graph : EpisodeGraph
            Current episode graph (used to build probeable_indices).
        deterministic : bool
            If True, select argmax action (greedy, for evaluation).
            If False, sample from the distribution (for training).

        Returns
        -------
        action   : int
        log_prob : float   log π(a|s)
        value    : float   V(s)
        entropy  : float   H(π(·|s))
        """
        # Build tensors from obs dict
        nf  = torch.tensor(obs_dict["node_features"], dtype=torch.float32)
        ei  = torch.tensor(obs_dict["edge_index"],    dtype=torch.int64)
        sh  = torch.tensor(obs_dict["slo_health"],    dtype=torch.float32)
        am  = torch.tensor(obs_dict["action_mask"],   dtype=torch.bool)

        nm, em = build_masks(episode_graph)
        pi     = build_probeable_indices(episode_graph)

        out = self.forward(nf, ei, nm, em, sh, am, pi)

        if deterministic:
            action = int(out.action_probs.argmax().item())
        else:
            dist   = Categorical(probs=out.action_probs)
            action = int(dist.sample().item())

        log_prob = float(out.log_probs[action].item())
        value    = float(out.state_value.item())
        entropy  = float(out.entropy.item())

        return action, log_prob, value, entropy