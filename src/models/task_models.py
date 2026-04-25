import torch.nn as nn
from src.models.gin_model import GraphPooling
from src.models.mlp_head import MLPHead


class Tox21Model(nn.Module):
    """Encoder + 12-task MLP head for Tox21 pretraining."""
    def __init__(self, encoder, hidden_dim=128, num_tasks=12, dropout_head=0.2):
        super().__init__()
        self.encoder = encoder
        self.pool = GraphPooling()
        self.head = MLPHead(hidden_dim, num_tasks, hidden_dim, dropout_head)

    def forward(self, x, edge_index, edge_attr, batch):
        node_emb = self.encoder(x, edge_index, edge_attr)
        graph_emb = self.pool(node_emb, batch)
        return self.head(graph_emb)


class NuBBEModel(nn.Module):
    """Same encoder but with a 1-task MLP head (binary antioxidant/ROS)."""
    def __init__(self, encoder, hidden_dim=128, dropout_head=0.2):
        super().__init__()
        self.encoder = encoder
        self.pool = GraphPooling()
        self.head = MLPHead(hidden_dim, 1, hidden_dim, dropout_head)

    def forward(self, x, edge_index, edge_attr, batch):
        node_emb = self.encoder(x, edge_index, edge_attr)
        graph_emb = self.pool(node_emb, batch)
        return self.head(graph_emb)