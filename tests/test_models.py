import torch
from src.models.gin_model import GINEEncoder

def test_gine_encoder_forward():
    model = GINEEncoder(in_channels=9, edge_dim=3)

    x = torch.randn(10, 9)
    edge_index = torch.randint(0, 10, (2, 20))
    edge_attr = torch.randn(20, 3)

    out = model(x, edge_index, edge_attr)

    print(f"Output shape: {out.shape}")          # e.g., torch.Size([10, 128])
    assert out.shape == (10, 128)
    print("Test passed! Output shape matches (10, 128)")

if __name__ == "__main__":
    test_gine_encoder_forward()
