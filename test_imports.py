import torch
import torch_geometric
import rdkit
import numpy as np
import pandas as pd

print("All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Geometric version: {torch_geometric.__version__}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Test basic tensor operations
x = torch.randn(3, 3)
print(f"Tensor test: {x.shape}")