# tests/test_models.py
import torch
import time
import numpy as np
from src.models.gin_model import GINEEncoder
from src.utils.graph_utils import build_knn_similarity_graph, build_knn_similarity_graph_faiss


def test_gine_encoder_forward():
    model = GINEEncoder(in_channels=9, edge_dim=3)
    x = torch.randn(10, 9)
    edge_index = torch.randint(0, 10, (2, 20))
    edge_attr = torch.randn(20, 3)

    out = model(x, edge_index, edge_attr)

    print(f"GINE output shape: {out.shape}")
    assert out.shape == (10, 128), f"Expected (10,128) but got {out.shape}"
    print("✓ GINEEncoder test passed!\n")


def test_knn_graph():
    emb = np.random.rand(50, 128)
    edge_index, N = build_knn_similarity_graph(emb, k=5)

    print(f"Number of nodes: {N}")
    print(f"Edge index shape: {edge_index.shape}")
    assert N == 50, f"Expected N=50 but got {N}"
    assert edge_index.shape[0] == 2, f"Edge index should have 2 rows, got {edge_index.shape[0]}"
    print("✓ k-NN graph test passed!\n")

def test_knn_graph_faiss():
    emb = np.random.rand(50, 128).astype(np.float32)  # Faiss requires float32
    edge_index, N = build_knn_similarity_graph_faiss(emb, k=5)
    print(f"[Faiss] Number of nodes: {N}")
    print(f"[Faiss] Edge index shape: {edge_index.shape}")
    assert N == 50
    assert edge_index.shape[0] == 2
    # Optionally check that at least one edge exists (with random embeddings, many may appear)
    print("✓ Faiss k-NN graph test passed!\n")

def test_knn_graph_faiss_large():
    print("Testing Faiss with 5000 random embeddings ...")
    t0 = time.time()
    emb = np.random.rand(5000, 128).astype(np.float32)
    edge_index, N = build_knn_similarity_graph_faiss(emb, k=10, threshold=0.6)
    t1 = time.time()
    print(f"[Faiss Large] Nodes: {N}")
    print(f"[Faiss Large] Edge index shape: {edge_index.shape}")
    print(f"[Faiss Large] Time: {t1 - t0:.2f} seconds")
    assert N == 5000
    assert edge_index.shape[0] == 2
    print("✓ Faiss large k-NN test passed!\n")


if __name__ == "__main__":
    #test_gine_encoder_forward()
    #test_knn_graph()
    #test_knn_graph_faiss()
    test_knn_graph_faiss_large()
    print("All tests finished successfully.")