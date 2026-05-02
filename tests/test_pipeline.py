from src.models.gin_model import GINEEncoder
from src.models.task_models import Tox21Model
import torch

def test_full_forward():
    print("Testing Tox21Model full forward pass...")

    encoder = GINEEncoder(in_channels=9, edge_dim=3)
    model = Tox21Model(encoder)

    x = torch.randn(10, 9)
    edge_index = torch.randint(0, 10, (2, 20))
    edge_attr = torch.randn(20, 3)
    batch = torch.zeros(10, dtype=torch.long)

    out = model(x, edge_index, edge_attr, batch)

    print(f"Output shape: {out.shape}")
    assert out.shape == (1, 12), f"Expected (1,12) but got {out.shape}"
    print("✓ Full forward pass test passed!\n")

if __name__ == "__main__":
    test_full_forward()