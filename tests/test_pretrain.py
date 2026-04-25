import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models.gin_model import GINEEncoder
from src.models.task_models import Tox21Model
from src.train.pretrain import pretrain_tox21

def generate_fake_graph():
    x = torch.randn(8, 9)                  # 8 atoms, 9 features
    edge_index = torch.randint(0, 8, (2, 16))
    edge_attr = torch.randn(16, 3)
    y = torch.randint(0, 2, (1, 12)).float()   # 12 tasks, shape (1,12)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def test_pretrain_loop():
    print("Starting pretraining loop test...")

    dataset = [generate_fake_graph() for _ in range(10)]
    loader = DataLoader(dataset, batch_size=2)

    encoder = GINEEncoder(in_channels=9, edge_dim=3)
    model = Tox21Model(encoder)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Use torch.device, not plain string
    device = torch.device("cpu")
    pretrain_tox21(model, loader, optimizer, device=device, epochs=1)

    print("✓ Pretraining loop test completed successfully!\n")

if __name__ == "__main__":
    test_pretrain_loop()