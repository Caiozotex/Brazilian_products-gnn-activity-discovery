import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from src.models.gin_model import GINEEncoder, GraphPooling
from src.utils.graph_utils import extract_embeddings
import os

# Load just a few graphs for testing
TOX21_DIR = "data/processed/graphs_tox21"
tox21_files = sorted(os.listdir(TOX21_DIR))[:10]  # Just 10 for testing

tox21_list = []
for fname in tox21_files:
    if fname.endswith('.pt'):
        with torch.serialization.safe_globals([torch_geometric.data.Data]):
            data = torch.load(os.path.join(TOX21_DIR, fname), weights_only=False)
        data.y = data.y.view(12)
        tox21_list.append(data)

print(f"Loaded {len(tox21_list)} test graphs")

# Create a small dataloader
loader = DataLoader(tox21_list, batch_size=4, shuffle=False)

# Load encoder
encoder = GINEEncoder(in_channels=35, edge_dim=9, hidden_channels=128, num_layers=4, dropout=0.2)
encoder.load_state_dict(torch.load("models/checkpoints/tox21/best_encoder.pt", map_location="cpu"))
encoder.eval()

device = torch.device('cpu')
encoder.to(device)

print("Testing extract_embeddings function...")

try:
    embeddings = extract_embeddings(encoder, loader, device)
    print(f"✅ Success! Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()