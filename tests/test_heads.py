import torch
from src.models.mlp_head import MLPHead

def test_mlp_head():
    head = MLPHead(128, 12)
    x = torch.randn(4, 128)

    out = head(x)

    print(f"Output shape: {out.shape}") 
    assert out.shape == (4, 12)
    print("Test passed! Output shape matches (4, 12)")

if __name__ == "__main__":
    test_mlp_head()
    