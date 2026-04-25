import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm


# ------------------------------
# 1. GNN Encoder (GINEConv)
# ------------------------------
class GINEEncoder(nn.Module):
    """
    4-layer GINEConv encoder with BatchNorm, ReLU, Dropout after each convolution.
    Returns node embeddings for each graph (num_nodes, hidden_dim).
    """
    def __init__(self, in_channels, edge_dim, hidden_channels=128, num_layers=4, dropout=0.2):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            # Input size for the MLP inside GINEConv
            in_dim = in_channels if i == 0 else hidden_channels
            nn_mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            conv = GINEConv(nn_mlp, edge_dim=edge_dim, train_eps=False)
            self.convs.append(conv)
            self.bns.append(BatchNorm(hidden_channels))

    def forward(self, x, edge_index, edge_attr):
        # x: (N, in_channels), edge_index: (2, E), edge_attr: (E, edge_dim)
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # node embeddings

# ------------------------------
# 2. Graph-Level Pooling and Embedding
#    (used inside the task models)
# ------------------------------
class GraphPooling(nn.Module):
    """Simple wrapper that applies global_mean_pool."""
    def forward(self, node_emb, batch):
        return global_mean_pool(node_emb, batch)