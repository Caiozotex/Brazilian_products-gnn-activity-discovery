"""
k-NN similarity graph using the same hybrid criterion as check_graph_connectivity.py:

    sim(i,j) = 0.2 * Tanimoto(Morgan fp)  +  0.8 * CenteredCosine(GNN emb)

Instead of a threshold, each node is connected to its k most similar neighbours.
This guarantees zero isolated nodes by construction.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

ALPHA     = 0.2
CHUNK     = 512
EMB_PATH  = "results/embeddings/joint_embeddings.npy"
META_PATH = "results/embeddings/joint_embeddings_meta.csv"
JOINT_CSV = "data/processed/joint_data.csv"

# ── Load embeddings (centered cosine) ─────────────────────────
print("Loading GNN embeddings ...")
emb = np.load(EMB_PATH).astype(np.float32)
N   = emb.shape[0]
cc  = emb - emb.mean(axis=0, keepdims=True)
cc /= np.where(np.linalg.norm(cc, axis=1, keepdims=True) == 0, 1.0,
               np.linalg.norm(cc, axis=1, keepdims=True))
print(f"  {N:,} compounds, dim={emb.shape[1]}")

# ── Load Morgan fingerprints ───────────────────────────────────
print("Loading Morgan fingerprints ...")
joint = pd.read_csv(JOINT_CSV)
meta  = pd.read_csv(META_PATH)
idx_to_smiles = joint.set_index("joint_idx")["smiles"].to_dict()
fps = []
for jidx in meta["joint_idx"]:
    smi = idx_to_smiles.get(jidx, None)
    mol = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) else None
    if mol is None:
        fps.append(np.zeros(2048, dtype=np.float32)); continue
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    ConvertToNumpyArray(fp, arr); fps.append(arr)
fps_mat = np.stack(fps)
cnt_all = fps_mat.sum(axis=1, keepdims=True)
print("  Done.\n")

# ── k-NN graph builder ─────────────────────────────────────────

def build_knn_graph(k):
    """
    For each node i, find its k nearest neighbours by hybrid similarity.
    Edges are undirected: if j is in top-k of i OR i is in top-k of j.
    """
    # top_k_neighbours[i] = sorted list of (sim, j) for node i's k best
    top_k = [[] for _ in range(N)]   # list of (sim, j) per node

    for s in range(0, N, CHUNK):
        e = min(s + CHUNK, N)

        # Hybrid sim block: shape (chunk, N)
        tan_chunk = fps_mat[s:e]
        inter     = tan_chunk @ fps_mat.T
        union     = cnt_all[s:e] + cnt_all.T - inter
        tan_block = np.where(union > 0, inter / union, 0.0)
        cos_block = cc[s:e] @ cc.T
        sim_block = ALPHA * tan_block + (1 - ALPHA) * cos_block

        # For each row (node i in chunk), pick top-k excluding self
        for li, gi in enumerate(range(s, e)):
            row = sim_block[li].copy()
            row[gi] = -np.inf                  # exclude self
            top_idx = np.argpartition(row, -k)[-k:]
            top_sims = row[top_idx]
            for j, sim_val in zip(top_idx.tolist(), top_sims.tolist()):
                top_k[gi].append((sim_val, j))

        if (s // CHUNK) % 20 == 0:
            print(f"  k={k}: {e}/{N} ({e/N*100:.1f}%)")

    # Build edge set (undirected: union of directed k-NN)
    edge_set = set()
    for i, neighbours in enumerate(top_k):
        for _, j in neighbours:
            a, b = (i, j) if i < j else (j, i)
            edge_set.add((a, b))

    ri = np.array([a for a, b in edge_set], dtype=np.int32)
    rj = np.array([b for a, b in edge_set], dtype=np.int32)
    n_edges = len(ri)

    data = np.ones(n_edges * 2, dtype=np.int8)
    r    = np.concatenate([ri, rj])
    c    = np.concatenate([rj, ri])
    adj  = sp.csr_matrix((data, (r, c)), shape=(N, N), dtype=np.int8)
    nc, lab = csgraph.connected_components(adj, directed=False)
    sz  = np.bincount(lab)

    return n_edges, nc, sz.max(), (sz == 1).sum()


# ── Sweep over k values ────────────────────────────────────────
print(f"Metric: hybrid sim = {ALPHA}*Tanimoto + {1-ALPHA}*CenteredCosine  (old GNN emb)")
print(f"{'k':>5}  {'edges':>10}  {'components':>12}  {'largest':>9}  {'isolated':>9}")
print("-" * 55)

for k in [5, 10, 15, 20]:
    n_edges, nc, largest, iso = build_knn_graph(k)
    print(f"{k:>5}  {n_edges:>10,}  {nc:>12,}  {largest:>9,}  {iso:>9,}")
