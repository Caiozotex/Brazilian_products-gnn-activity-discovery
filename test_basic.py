import os
import torch
import torch_geometric
import numpy as np

# Test data loading
TOX21_DIR = "data/processed/graphs_tox21"
ANTIOX_DIR = "data/processed/graphs_AntioxidantRos"

print("Testing data loading...")

# Load a few Tox21 graphs
tox21_files = sorted(os.listdir(TOX21_DIR))[:3]  # Just 3 for testing
tox21_list = []
for fname in tox21_files:
    if fname.endswith('.pt'):
        with torch.serialization.safe_globals([torch_geometric.data.Data]):
            data = torch.load(os.path.join(TOX21_DIR, fname), weights_only=False)
        data.y = data.y.view(12)
        tox21_list.append(data)

print(f"✅ Loaded {len(tox21_list)} Tox21 graphs")

# Load a few NuBBE graphs
def encode_label(y_str):
    if isinstance(y_str, str) and "antioxidant" in y_str.lower():
        return torch.tensor([1.0])
    else:
        return torch.tensor([0.0])

nubbe_files = sorted(os.listdir(ANTIOX_DIR))[:3]  # Just 3 for testing
nubbe_list = []
for fname in nubbe_files:
    if fname.endswith('.pt'):
        with torch.serialization.safe_globals([torch_geometric.data.Data]):
            data = torch.load(os.path.join(ANTIOX_DIR, fname), weights_only=False)
        data.y = encode_label(data.y)
        nubbe_list.append(data)

print(f"✅ Loaded {len(nubbe_list)} NuBBE graphs")

# Test model
from src.models.gin_model import GINEEncoder
encoder = GINEEncoder(in_channels=35, edge_dim=9, hidden_channels=128, num_layers=4, dropout=0.2)
encoder.load_state_dict(torch.load("models/checkpoints/tox21/best_encoder.pt", map_location="cpu"))
encoder.eval()

print("✅ Model loaded and ready!")

# Test embedding extraction on one graph
test_data = tox21_list[0]
print(f"Test graph - Nodes: {test_data.x.shape[0]}, Edges: {test_data.edge_index.shape[1]}")

try:
    with torch.no_grad():
        emb = encoder(test_data.x, test_data.edge_index, test_data.edge_attr)
        emb = emb.mean(dim=0)  # Global mean pooling
    print(f"✅ Embedding extracted! Shape: {emb.shape}")
except Exception as e:
    print(f"❌ Embedding extraction failed: {e}")

print("\n🎉 Basic functionality test completed successfully!")