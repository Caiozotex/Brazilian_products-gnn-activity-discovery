import os
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.data import Data
import pandas as pd
import torch_geometric

from src.models.gin_model import GINEEncoder
from src.models.task_models import Tox21Model, NuBBEModel
from src.train.pretrain import pretrain_tox21
from src.train.finetune import finetune_nubbe
from src.utils.graph_utils import extract_embeddings,build_knn_similarity_graph_faiss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Directories where individual .pt files live
    TOX21_DIR = "data/processed/graphs_tox21"
    ANTIOX_DIR = "data/processed/graphs_AntioxidantRos"

    # -------------------------------------------------------
    # Load ALL graphs into a list
    # -------------------------------------------------------
    # Tox21
    tox21_list = []
    for fname in sorted(os.listdir(TOX21_DIR)):
        if fname.endswith('.pt'):
            with torch.serialization.safe_globals([torch_geometric.data.Data]):
                data = torch.load(os.path.join(TOX21_DIR, fname), weights_only=False)
            # Ensure y is 1‑D with 12 elements
            data.y = data.y.view(12)          # reshape to (12,)
            tox21_list.append(data)

    # NuBBE (AntioxidantRos)
    def encode_label(y_str):
        """Convert bioactivity string to binary tensor."""
        if isinstance(y_str, str) and "antioxidant" in y_str.lower():
            return torch.tensor([1.0])
        else:
            return torch.tensor([0.0])
    
    nubbe_list = []
    for fname in sorted(os.listdir(ANTIOX_DIR)):
        if fname.endswith('.pt'):
            with torch.serialization.safe_globals([torch_geometric.data.Data]):
                data = torch.load(os.path.join(ANTIOX_DIR, fname), weights_only=False)
            # Convert string label to binary tensor (shape (1,))
            data.y = encode_label(data.y)
            nubbe_list.append(data)

    print(f"Tox21 graphs: {len(tox21_list)}")
    print(f"NuBBE (AntioxidantRos) graphs: {len(nubbe_list)}")


    # -------------------------------------------------------
    # Create DataLoaders
    # -------------------------------------------------------
    tox21_loader = DataLoader(tox21_list, batch_size=32, shuffle=True)
    nubbe_loader = DataLoader(nubbe_list, batch_size=32, shuffle=True)

    # dimensions
    num_node_feats = 35
    num_edge_feats = 9

    print("Loading encoder...")
    # Option A: Tox21‑pretrained encoder
    encoder = GINEEncoder(in_channels=num_node_feats,edge_dim=num_edge_feats, hidden_channels=128, num_layers=4, dropout=0.2).to(device)
    encoder.load_state_dict(torch.load("models/checkpoints/tox21/best_encoder.pt", map_location="cpu"))

    # Option B: NuBBE fine‑tuned encoder (if you fine‑tuned)
    # encoder.load_state_dict(torch.load("models/checkpoints/nubbe_antioxidant/best_encoder.pt", map_location="cpu"))

    encoder.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    print("Extracting embeddings...")
    # # 4. Extract embeddings for all molecules
    tox21_emb   = extract_embeddings(encoder, tox21_loader, device)
    print(f"Tox21 embeddings shape: {tox21_emb.shape}")
    nubbe_emb   = extract_embeddings(encoder, nubbe_loader, device)
    print(f"NuBBE embeddings shape: {nubbe_emb.shape}")
    assert np.all(np.isfinite(tox21_emb)), "Tox21 embeddings contain NaN or Inf"
    assert np.all(np.isfinite(nubbe_emb)), "NuBBE embeddings contain NaN or Inf"
    n_tox = tox21_emb.shape[0]          # number of Tox21 molecules
    n_nub = nubbe_emb.shape[0]          # number of NuBBE molecules
    all_emb = np.concatenate([tox21_emb, nubbe_emb], axis=0)
    num_nodes = all_emb.shape[0]

    print(f"Total embeddings: {num_nodes} (Tox21: {n_tox}, NuBBE: {n_nub})")

    # Skip full similarity matrix computation (too memory intensive)
    # Go directly to k-NN graph building
    print("Building k-NN similarity graph...")
    edge_index, num_nodes, edge_similarities = build_knn_similarity_graph_faiss(all_emb, k=10, threshold=0.95)
    print(f"Graph built: {num_nodes} nodes, {edge_index.shape[1]} edges")

    # Save edge similarities to CSV
    print("Saving edge similarities to CSV...")
    similarity_data = []
    for (node1, node2), sim_value in edge_similarities.items():
        similarity_data.append({'node1': node1, 'node2': node2, 'similarity': sim_value})
    
    similarity_df = pd.DataFrame(similarity_data)
    csv_dir = "results/similiarity_graph"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "edge_similarities.csv")
    similarity_df.to_csv(csv_path, index=False)
    print(f"✅ Edge similarities saved to {csv_path}")
    print(f"   CSV contains {len(similarity_data)} edges with their similarity values")

    # Remove duplicate undirected edges (keep i < j)
    mask = edge_index[0] < edge_index[1]
    undirected_edge_index = edge_index[:, mask]

    # 2. Optional: verify it's roughly half
    print(f"Directed edges (both dirs): {edge_index.shape[1]}")
    print(f"Undirected edges (unique): {undirected_edge_index.shape[1]}")

    # # # Create a node attribute that indicates the dataset:
    # # # 0 = Tox21, 1 = NuBBE
    node_dataset = torch.zeros(num_nodes, dtype=torch.long)
    node_dataset[n_tox : n_tox + n_nub] = 1   # indices after Tox21 are NuBBE

    # # # The resulting graph can be wrapped as a PyG Data object:
    similarity_graph_undirected = Data(
    edge_index=undirected_edge_index,
    num_nodes=num_nodes,
    node_dataset=node_dataset)

    save_dir = "results/similiarity_graph"
    save_path = os.path.join(save_dir, "undirected_knn_graph.pt")
    torch.save(similarity_graph_undirected, save_path)
    print(f"Undirected graph saved to {save_path}")

