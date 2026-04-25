import torch.nn as nn


# ------------------------------
# 3. Multi-Task Heads and Models
# ------------------------------
class MLPHead(nn.Module):
    """Two-layer MLP head."""
    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)