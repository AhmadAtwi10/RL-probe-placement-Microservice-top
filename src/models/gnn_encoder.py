"""
gnn_encoder.py
--------------
Graph Neural Network encoder for the probe placement POMDP.

Implements L layers of message passing over the episode graph G^(e),
producing per-node embeddings h_v and a global graph embedding z
used by the policy network.

─────────────────────────────────────────────────────────────────
Architecture
─────────────────────────────────────────────────────────────────

Each GNN layer performs:

    m_v(l) = AGGREGATE({ W_msg · h_u(l-1) | u ∈ N_in(v) })
    h_v(l) = ReLU( LayerNorm( W_upd · concat(h_v(l-1), m_v(l)) ) )

where N_in(v) is the set of in-neighbours of v in G^(e).

After L layers, global readout via masked mean pooling:

    z = (1 / |V^(e)|) · Σ_{v ∈ V^(e)} h_v(L)

The node mask (True for real nodes, False for padding) ensures
padded rows do not contaminate the mean.

─────────────────────────────────────────────────────────────────
Stress signal propagation (§6 of formulation)
─────────────────────────────────────────────────────────────────
The GNN propagates stress signals from probed to unprobed neighbours.
If search is probed and shows rising latency, after 1 message-passing
step geo (its dependency) will have received that signal in its
aggregated messages, even though geo is unprobed.  This is why L >= 2
is important: the signal can travel multiple hops.

─────────────────────────────────────────────────────────────────
Variable graph size handling
─────────────────────────────────────────────────────────────────
- node_features is padded to (MAX_NODES=24, NODE_FEATURE_DIM=62)
- node_mask    is bool (MAX_NODES,): True = real, False = padding
- edge_index   is padded to (2, MAX_EDGES=41): padded cols are 0s
  and are excluded via edge_mask (True only for real edges)

Weight matrices are shared across all nodes and all episodes —
the GNN generalises zero-shot to any subgraph of G^(e).

─────────────────────────────────────────────────────────────────
Public API
─────────────────────────────────────────────────────────────────
  GNNConfig            — architecture hyperparameters
  GNNEncoder(config)   — nn.Module
    .forward(node_features, edge_index, node_mask, edge_mask)
      → node_embeddings (MAX_NODES, hidden_dim)
         graph_embedding (hidden_dim,)

  build_masks(episode_graph, max_nodes, max_edges)
      → node_mask (MAX_NODES,), edge_mask (MAX_EDGES,)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.simulator.env.observation_builder import NODE_FEATURE_DIM

# Maximum graph sizes (fixed across all episodes)
MAX_NODES: int = 24
MAX_EDGES: int = 200   # matches env observation_space edge_index padding


# ---------------------------------------------------------------------------
# GNNConfig
# ---------------------------------------------------------------------------

@dataclass
class GNNConfig:
    """
    Architecture hyperparameters for the GNN encoder.

    Attributes
    ----------
    input_dim : int
        Node feature dimension.  Must match NODE_FEATURE_DIM (62).
    hidden_dim : int
        Dimension of node embeddings at each GNN layer.
    num_layers : int
        L — number of message-passing layers.
        L=2 allows stress signals to travel 2 hops (e.g. frontend→search→geo).
        L=3 recommended for full DeathStarBench depth.
    dropout : float
        Dropout probability applied after each layer (0 = disabled).
    """
    input_dim:  int   = NODE_FEATURE_DIM   # 62
    hidden_dim: int   = 128
    num_layers: int   = 3
    dropout:    float = 0.1

    def __post_init__(self):
        assert self.input_dim  > 0
        assert self.hidden_dim > 0
        assert self.num_layers >= 1
        assert 0.0 <= self.dropout < 1.0


# ---------------------------------------------------------------------------
# GNNLayer — single message-passing step
# ---------------------------------------------------------------------------

class GNNLayer(nn.Module):
    """
    One message-passing layer.

    Message: W_msg applied to source node embeddings.
    Update:  W_upd applied to concat(self, aggregated_messages),
             followed by LayerNorm and ReLU.

    Parameters
    ----------
    in_dim : int
        Input node embedding dimension.
    out_dim : int
        Output node embedding dimension (= hidden_dim).
    dropout : float
        Dropout probability.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W_msg  = nn.Linear(in_dim, out_dim, bias=False) # Its a learnable linear layer used to transform the node source embedding before sending as a message
        self.W_upd  = nn.Linear(in_dim + out_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)
        self.drop   = nn.Dropout(dropout)

    def forward(
        self,
        h:          torch.Tensor,   # (N, in_dim)  current node embeddings
        edge_index: torch.Tensor,   # (2, E)       COO edge index (int64)
        edge_mask:  torch.Tensor,   # (E,)         bool — real vs padded edges
        node_mask:  torch.Tensor,   # (N,)         bool — real vs padded nodes
    ) -> torch.Tensor:              # (N, out_dim)
        """
        Parameters
        ----------
        h : (N, in_dim)
        edge_index : (2, E) — row 0 = src, row 1 = dst
        edge_mask  : (E,)   — True for real edges
        node_mask  : (N,)   — True for real nodes

        Returns
        -------
        h_new : (N, out_dim)
        """
        N = h.size(0) # Gives nb of nodes 

        # ── Message: transform source node features ──────────────────────
        src_idx = edge_index[0]   # (E,)
        dst_idx = edge_index[1]   # (E,)

        # msg shape: (E, out_dim)
        msg = self.W_msg(h[src_idx])

        # Zero out messages from padded edges
        msg = msg * edge_mask.unsqueeze(-1).float()

        # ── Aggregate: sum messages at each destination node ─────────────
        # shape: (N, out_dim)
        agg = torch.zeros(N, msg.size(-1), dtype=h.dtype, device=h.device)
        agg.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(msg), msg)

        # Normalise by in-degree (count of real incoming edges per node)
        # to make aggregation mean rather than sum — more stable gradients
        deg = torch.zeros(N, dtype=h.dtype, device=h.device)
        deg.scatter_add_(0, dst_idx, edge_mask.float())
        deg = deg.clamp(min=1.0)   # avoid divide-by-zero for isolated nodes
        agg = agg / deg.unsqueeze(-1)

        # ── Update: concat self + aggregated, project, normalise ─────────
        combined = torch.cat([h, agg], dim=-1)   # (N, in_dim + out_dim)
        h_new    = self.W_upd(combined)           # (N, out_dim)
        h_new    = self.norm(h_new)
        h_new    = F.relu(h_new)
        h_new    = self.drop(h_new)

        # Zero out padded node rows to prevent them from leaking into
        # subsequent layers via their embeddings
        h_new = h_new * node_mask.unsqueeze(-1).float()

        return h_new


# ---------------------------------------------------------------------------
# GNNEncoder
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    """
    Multi-layer GNN encoder for the episode graph.

    Takes padded node features and graph structure, produces per-node
    embeddings and a global graph embedding for the policy network.

    Parameters
    ----------
    config : GNNConfig
        Architecture hyperparameters.
    """

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config

        # Input projection: NODE_FEATURE_DIM → hidden_dim
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # L message-passing layers (all hidden_dim → hidden_dim after layer 0)
        self.layers = nn.ModuleList([
            GNNLayer(
                in_dim  = config.hidden_dim,
                out_dim = config.hidden_dim,
                dropout = config.dropout,
            )
            for _ in range(config.num_layers)
        ])

    def forward(
        self,
        node_features: torch.Tensor,   # (B, N, input_dim) or (N, input_dim)
        edge_index:    torch.Tensor,   # (B, 2, E) or (2, E)
        node_mask:     torch.Tensor,   # (B, N) or (N,)   bool
        edge_mask:     torch.Tensor,   # (B, E) or (E,)   bool
    ):
        """
        Forward pass.

        Supports both batched (B, N, ...) and unbatched (N, ...) inputs.
        Unbatched inputs are temporarily given a batch dimension internally.

        Parameters
        ----------
        node_features : (B, N, input_dim) or (N, input_dim)
        edge_index    : (B, 2, E)         or (2, E)   int64
        node_mask     : (B, N)            or (N,)     bool
        edge_mask     : (B, E)            or (E,)     bool

        Returns
        -------
        node_embeddings : same batch shape as input, (B, N, hidden_dim) or (N, hidden_dim)
            Per-node embeddings after L message-passing layers.
            Padded node rows are zeroed out.
        graph_embedding : (B, hidden_dim) or (hidden_dim,)
            Global graph embedding via masked mean pooling over real nodes.
        """
        unbatched = node_features.dim() == 2
        if unbatched:
            node_features = node_features.unsqueeze(0)   # (1, N, D)
            edge_index    = edge_index.unsqueeze(0)       # (1, 2, E)
            node_mask     = node_mask.unsqueeze(0)        # (1, N)
            edge_mask     = edge_mask.unsqueeze(0)        # (1, E)

        B, N, _ = node_features.shape
        E       = edge_index.shape[-1]

        # ── Input projection ─────────────────────────────────────────────
        # (B, N, hidden_dim)
        h = self.input_proj(node_features)
        h = F.relu(h)

        # Zero padded nodes immediately after projection
        h = h * node_mask.unsqueeze(-1).float()

        # ── Message passing layers ────────────────────────────────────────
        # Process each item in the batch independently through each layer.
        # (No cross-item edges exist — batch items are independent graphs.)
        for layer in self.layers:
            h_next = torch.zeros_like(h)
            for b in range(B):
                h_next[b] = layer(
                    h          = h[b],
                    edge_index = edge_index[b],
                    edge_mask  = edge_mask[b],
                    node_mask  = node_mask[b],
                )
            h = h_next

        # ── Global readout: masked mean pooling ───────────────────────────
        # Sum embeddings of real nodes, divide by real node count
        # node_mask: (B, N) → (B, N, 1) for broadcasting
        mask_f       = node_mask.float().unsqueeze(-1)          # (B, N, 1)
        node_sum     = (h * mask_f).sum(dim=1)                  # (B, hidden_dim)
        node_count   = mask_f.sum(dim=1).clamp(min=1.0)         # (B, 1)
        graph_embed  = node_sum / node_count                    # (B, hidden_dim)

        # ── Remove batch dim if input was unbatched ───────────────────────
        if unbatched:
            h           = h.squeeze(0)           # (N, hidden_dim)
            graph_embed = graph_embed.squeeze(0)  # (hidden_dim,)

        return h, graph_embed


# ---------------------------------------------------------------------------
# Helper: build masks from EpisodeGraph
# ---------------------------------------------------------------------------

def build_masks(
    episode_graph,
    max_nodes: int = MAX_NODES,
    max_edges: int = MAX_EDGES,
) -> tuple:
    """
    Build node_mask and edge_mask tensors from an EpisodeGraph.

    Parameters
    ----------
    episode_graph : EpisodeGraph
        The active episode graph.
    max_nodes : int
        Padded node dimension (default MAX_NODES=24).
    max_edges : int
        Padded edge dimension (default MAX_EDGES=200, matches env padding).

    Returns
    -------
    node_mask : torch.BoolTensor  (max_nodes,)
        True for real nodes, False for padding rows.
    edge_mask : torch.BoolTensor  (max_edges,)
        True for real edges, False for padding columns.
    """
    n_real = episode_graph.num_nodes
    _, (src_list, _) = episode_graph.get_edge_index()
    e_real = len(src_list)   # real edge count

    node_mask = torch.zeros(max_nodes, dtype=torch.bool)
    node_mask[:n_real] = True

    edge_mask = torch.zeros(max_edges, dtype=torch.bool)
    edge_mask[:e_real] = True

    return node_mask, edge_mask