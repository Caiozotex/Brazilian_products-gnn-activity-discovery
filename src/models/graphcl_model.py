"""
GraphCL: Graph Contrastive Learning components.

Architecture:
    GINEEncoder (backbone, shared with downstream tasks)
        -> global_mean_pool
        -> ProjectionHead (MLP 128->128->64, used only during contrastive training)
        -> L2 normalize
    NT-Xent loss (InfoNCE) over augmented view pairs.

Reference: You et al., "Graph Contrastive Learning with Augmentations" (NeurIPS 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class ProjectionHead(nn.Module):
    """Small MLP that maps graph embeddings to the contrastive projection space."""

    def __init__(self, in_dim: int = 128, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GraphCLModel(nn.Module):
    """
    Wraps a GINEEncoder with a projection head for contrastive training.

    After training, discard the projection head and use only `encoder`
    to extract embeddings for downstream tasks.
    """

    def __init__(self, encoder: nn.Module, proj_hidden: int = 128, proj_out: int = 64):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(
            in_dim=encoder.hidden_channels,
            hidden_dim=proj_hidden,
            out_dim=proj_out,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        node_emb  = self.encoder(x, edge_index, edge_attr)       # (N_nodes, 128)
        graph_emb = global_mean_pool(node_emb, batch)             # (B, 128)
        z         = self.projector(graph_emb)                     # (B, 64)
        z         = F.normalize(z, dim=1)                         # unit sphere
        return z


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    Given two batches of projections z_i and z_j (one per augmented view),
    treats (z_i[k], z_j[k]) as a positive pair and all other 2(B-1) samples
    as negatives.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        z_i, z_j: (B, D) — already L2-normalized.
        Returns scalar loss.
        """
        B      = z_i.size(0)
        device = z_i.device

        # Concatenate: first B rows are view-i, next B are view-j  -> (2B, D)
        z   = torch.cat([z_i, z_j], dim=0)

        # Pairwise cosine similarities (already normalized, so = dot product)
        sim = (z @ z.T) / self.temperature   # (2B, 2B)

        # Mask out self-similarities (diagonal)
        mask = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim  = sim.masked_fill(mask, float("-inf"))

        # Positive pair for row i is row i+B (and vice-versa)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=device),
            torch.arange(0, B,     device=device),
        ])   # (2B,)

        loss = F.cross_entropy(sim, labels)
        return loss
