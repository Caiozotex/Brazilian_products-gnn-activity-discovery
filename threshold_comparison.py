import os
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.data import Data
import pandas as pd
import torch_geometric
import networkx as nx

from src.models.gin_model import GINEEncoder
from src.utils.graph_utils import extract_embeddings, build_knn_similarity_graph_faiss


def analyze_threshold(threshold, all_emb, num_nodes):
    """Analyze graph metrics for a given threshold."""
    print(f"\n--- Analyzing threshold {threshold} ---")

    # Build k-NN similarity graph
    edge_index, _, _ = build_knn_similarity_graph_faiss(all_emb, k=10, threshold=threshold)

    # Calculate number of edges (undirected)
    num_edges = edge_index.shape[1] // 2  # Divide by 2 since edges are stored in both directions
    print(f"Number of edges: {num_edges}")

    # Calculate average degree
    avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
    print(f"Average degree: {avg_degree:.2f}")

    # Calculate connected components
    # Convert to NetworkX graph for connected components analysis
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    num_components = nx.number_connected_components(G)
    print(f"Number of connected components: {num_components}")

    return {
        'threshold': threshold,
        'num_edges': num_edges,
        'avg_degree': round(avg_degree, 2),
        'num_components': num_components
    }


def main():
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
    encoder = GINEEncoder(in_channels=num_node_feats, edge_dim=num_edge_feats, hidden_channels=128, num_layers=4, dropout=0.2).to(device)
    encoder.load_state_dict(torch.load("models/checkpoints/tox21/best_encoder.pt", map_location="cpu"))

    encoder.eval()
    encoder.to(device)

    print("Extracting embeddings...")
    # Extract embeddings for all molecules
    tox21_emb = extract_embeddings(encoder, tox21_loader, device)
    print(f"Tox21 embeddings shape: {tox21_emb.shape}")
    nubbe_emb = extract_embeddings(encoder, nubbe_loader, device)
    print(f"NuBBE embeddings shape: {nubbe_emb.shape}")

    assert np.all(np.isfinite(tox21_emb)), "Tox21 embeddings contain NaN or Inf"
    assert np.all(np.isfinite(nubbe_emb)), "NuBBE embeddings contain NaN or Inf"

    n_tox = tox21_emb.shape[0]          # number of Tox21 molecules
    n_nub = nubbe_emb.shape[0]          # number of NuBBE molecules
    all_emb = np.concatenate([tox21_emb, nubbe_emb], axis=0)
    num_nodes = all_emb.shape[0]

    print(f"Total embeddings: {num_nodes} (Tox21: {n_tox}, NuBBE: {n_nub})")

    # Test different thresholds
    thresholds = [0.6, 0.8, 0.82, 0.83, 0.85, 0.9, 0.95]
    results = []

    for threshold in thresholds:
        result = analyze_threshold(threshold, all_emb, num_nodes)
        results.append(result)

    # Print comparison table
    print("\n" + "="*60)
    print("THRESHOLD COMPARISON TABLE")
    print("="*60)
    print(f"{'Threshold':<12} {'Edges':<12} {'Avg Degree':<12} {'Components':<12}")
    print("-" * 60)

    for result in results:
        print(f"{result['threshold']:<12} {result['num_edges']:<12} {result['avg_degree']:<12} {result['num_components']:<12}")

    print("="*60)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_path = "results/threshold_comparison.csv"
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()