import torch
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import faiss

from src.models.gin_model import GraphPooling



# ------------------------------
# 7. Embedding Extraction
# ------------------------------
@torch.no_grad()
def extract_embeddings(encoder, dataloader, device):
    """
    Returns a (num_molecules, 128) numpy array of graph embeddings.
    `encoder` can be any module that takes (x, edge_index, edge_attr) and returns node embeddings.
    """
    encoder.eval()
    pool = GraphPooling()
    embeddings = []
    for batch in dataloader:
        batch = batch.to(device)
        node_emb = encoder(batch.x, batch.edge_index, batch.edge_attr)
        graph_emb = pool(node_emb, batch.batch)
        embeddings.append(graph_emb.cpu())
    all_emb = torch.cat(embeddings, dim=0)
    return all_emb.numpy()

# ------------------------------
# 8. Cosine Similarity & k-NN Graph
# ------------------------------
def build_knn_similarity_graph(embeddings, k=10, threshold=0.6):
    """
    embeddings: (num_molecules, 128) numpy array.
    Returns:
        edge_index: torch.LongTensor of shape (2, E) – undirected edges,
        num_nodes: int
    The graph contains an edge between i and j if:
        - j is among the k nearest neighbours of i (or vice‑versa) in cosine space,
        - cosine similarity >= threshold.
    """
    # L2-normalize so that dot product = cosine similarity
    emb_norm = normalize(embeddings, norm='l2')  # (N, D)
    num_nodes = emb_norm.shape[0]

    # Use sklearn's NearestNeighbors for efficient k-NN
    nn_model = NearestNeighbors(n_neighbors=k+1, metric='cosine')  # cosine distance
    nn_model.fit(emb_norm)
    distances, indices = nn_model.kneighbors(emb_norm)  # distances are cosine distances

    # Convert cosine distance to similarity: cosine_sim = 1 - cosine_distance
    similarities = 1 - distances  # (N, k+1), column 0 = self (sim=1)

    edge_list = []
    for i in range(num_nodes):
        for j_idx in range(1, k+1):  # skip self (index 0)
            j = indices[i, j_idx]
            sim = similarities[i, j_idx]
            if sim >= threshold:
                # add directed edge (i, j); we'll symmetrize later
                edge_list.append([i, j])

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # Make undirected: add reverse edges, then remove duplicates
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index = torch.unique(edge_index, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return edge_index, num_nodes

def build_knn_similarity_graph_faiss(embeddings, k=10, threshold=0.6):
    """
    Use Faiss (cosine similarity -> inner product on L2-normalized vectors).
    Much faster and memory-friendly for large N.
    
    Returns:
        edge_index: torch tensor of shape (2, E)
        num_nodes: int
        edge_similarities: dict with (i, j) tuples as keys and similarity values
    """
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    # L2-normalize so inner product = cosine similarity
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    n = emb.shape[0]
    
    # Build index
    index = faiss.IndexFlatIP(emb.shape[1])   # inner product
    index.add(emb)
    
    # Search for k+1 neighbours (the first hit will be the point itself)
    similarities, indices = index.search(emb, k + 1)
    
    edge_list = []
    edge_similarities = {}  # Store similarity for each edge
    for i in range(n):
        for j_idx, sim in zip(indices[i, 1:], similarities[i, 1:]):  # skip self
            if sim >= threshold and i != j_idx:  # Ensure no self-loops
                edge_list.append([i, j_idx])
                # Store as undirected edge (always smaller index first)
                edge_key = (min(i, j_idx), max(i, j_idx))
                # Keep the maximum similarity if edge appears twice
                if edge_key not in edge_similarities:
                    edge_similarities[edge_key] = float(sim)
                else:
                    edge_similarities[edge_key] = max(edge_similarities[edge_key], float(sim))
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    # Make undirected and unique
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    
    return edge_index, n, edge_similarities


def build_threshold_similarity_graph(embeddings, threshold=0.6):
    """
    Build a similarity graph by comparing every pair of nodes.
    No k-NN search is used. An undirected edge is created for each
    pair whose cosine similarity is greater than or equal to the threshold.

    Returns:
        edge_index: torch tensor of shape (2, E)
        num_nodes: int
        edge_similarities: dict with (i, j) keys and cosine similarity values
    """
    emb_norm = normalize(embeddings, norm='l2')
    num_nodes = emb_norm.shape[0]

    sim_matrix = emb_norm @ emb_norm.T
    i_upper, j_upper = np.triu_indices(num_nodes, k=1)
    sim_values = sim_matrix[i_upper, j_upper]
    valid = sim_values >= threshold

    edge_similarities = {}
    if valid.any():
        i_valid = i_upper[valid]
        j_valid = j_upper[valid]
        sim_valid = sim_values[valid]

        for i, j, sim in zip(i_valid, j_valid, sim_valid):
            edge_similarities[(int(i), int(j))] = float(sim)

        edge_list = np.vstack([np.concatenate([i_valid, j_valid]), np.concatenate([j_valid, i_valid])])
        edge_index = torch.tensor(edge_list, dtype=torch.long)
        edge_index = torch.unique(edge_index, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return edge_index, num_nodes, edge_similarities