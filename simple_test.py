import os
# Disable CUDA before any torch imports
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_USE_CUDA_DSA'] = '0'

import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Test basic tensor operations
x = torch.randn(3, 3)
print("Tensor shape:", x.shape)
print("Basic operations work!")

# Try importing torch_geometric
try:
    import torch_geometric
    print("PyTorch Geometric version:", torch_geometric.__version__)
except Exception as e:
    print("Error importing torch_geometric:", e)

print("✅ Basic imports successful!")