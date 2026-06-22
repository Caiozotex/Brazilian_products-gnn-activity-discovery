"""
Build similarity graph (hybrid alpha=0.2, threshold=0.84) and check connectivity.

Nodes  : each row in joint_embeddings_meta.csv (one per compound)
Edges  : pairs with hybrid similarity >= THRESHOLD
         hybrid = ALPHA * tanimoto(fp) + (1-ALPHA) * centered_cosine(emb)

Reports:
  - number of nodes and edges
  - number of connected components
  - size of largest component
  - isolated nodes (degree 0)
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
import warnings
warnings.filterwarnings("ignore")

ALPHA     = 0.2
THRESHOLD = 0.84
CHUNK     = 256

EMB_PATH  = "results/embeddings/joint_embeddings.npy"
META_PATH = "results/embeddings/joint_embeddings_meta.csv"
JOINT_CSV = "data/processed/joint_data.csv"

# ──────────────────────────────────────────────────────────────
# Load features
# ──────────────────────────────────────────────────────────────
print("Loading centered cosine embeddings ...")
emb  = np.load(EMB_PATH).astype(np.float32)
emb  = emb - emb.mean(axis=0, keepdims=True)
norms = np.linalg.norm(emb, axis=1, keepdims=True)
emb  = emb / np.where(norms == 0, 1.0, norms)
N    = emb.shape[0]
print(f"  {N} compounds")

print("Loading Morgan fingerprints ...")
joint = pd.read_csv(JOINT_CSV)
meta  = pd.read_csv(META_PATH)
idx_to_smiles = joint.set_index("joint_idx")["smiles"].to_dict()

fps, valid = [], []
for jidx in meta["joint_idx"]:
    smi = idx_to_smiles.get(jidx, None)
    mol = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) else None
    if mol is None:
        fps.append(np.zeros(2048, dtype=np.float32)); valid.append(False); continue
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    fps.append(arr); valid.append(True)

fps_mat = np.stack(fps)
n_invalid = sum(not v for v in valid)
if n_invalid:
    print(f"  ! {n_invalid} invalid SMILES (zero fingerprint used)")
print(f"  fingerprints ready\n")

# ──────────────────────────────────────────────────────────────
# Build edge list
# ──────────────────────────────────────────────────────────────
print(f"Building edges (alpha={ALPHA}, threshold={THRESHOLD}) ...")

rows_i, rows_j, weights = [], [], []

for chunk_start in range(0, N, CHUNK):
    chunk_end   = min(chunk_start + CHUNK, N)
    cc_chunk    = emb[chunk_start:chunk_end]
    tan_chunk   = fps_mat[chunk_start:chunk_end]

    # centered cosine block
    cc_block = cc_chunk @ emb.T                                       # (B, N)

    # tanimoto block
    inter    = tan_chunk @ fps_mat.T                                  # (B, N)
    cnt_c    = tan_chunk.sum(axis=1, keepdims=True)
    cnt_m    = fps_mat.sum(axis=1, keepdims=True).T
    union    = cnt_c + cnt_m - inter
    tan_block = np.where(union > 0, inter / union, 0.0)

    sim_block = ALPHA * tan_block + (1 - ALPHA) * cc_block            # (B, N)

    for local_i, global_i in enumerate(range(chunk_start, chunk_end)):
        sim_row = sim_block[local_i, global_i + 1:]                   # upper triangle
        js      = np.where(sim_row >= THRESHOLD)[0] + global_i + 1

        if len(js) == 0:
            continue

        rows_i.extend([global_i] * len(js))
        rows_j.extend(js.tolist())
        weights.extend(sim_row[js - global_i - 1].tolist())

    if (chunk_start // CHUNK) % 20 == 0:
        pct = chunk_end / N * 100
        print(f"  {chunk_end}/{N} ({pct:.1f}%)  edges so far: {len(rows_i):,}")

print(f"\nTotal edges (undirected): {len(rows_i):,}")

# ──────────────────────────────────────────────────────────────
# Build sparse adjacency and check connectivity
# ──────────────────────────────────────────────────────────────
print("\nBuilding sparse adjacency matrix ...")
data = np.ones(len(rows_i) * 2, dtype=np.float32)
row  = np.array(rows_i + rows_j, dtype=np.int32)
col  = np.array(rows_j + rows_i, dtype=np.int32)
adj  = sp.csr_matrix((data, (row, col)), shape=(N, N))

print("Checking connectivity ...")
n_components, labels = csgraph.connected_components(adj, directed=False)

component_sizes = np.bincount(labels)
largest         = component_sizes.max()
isolated        = (component_sizes == 1).sum()
in_main         = (labels == np.argmax(component_sizes)).sum()

print(f"\n{'='*45}")
print(f"  Nodes              : {N:,}")
print(f"  Edges (undirected) : {len(rows_i):,}")
print(f"  Connected          : {'YES' if n_components == 1 else 'NO'}")
print(f"  Components         : {n_components:,}")
print(f"  Largest component  : {largest:,} nodes ({largest/N*100:.1f}%)")
print(f"  Isolated nodes     : {isolated:,} ({isolated/N*100:.2f}%)")
if n_components > 1:
    sizes_sorted = sorted(component_sizes, reverse=True)
    print(f"  Top-5 component sizes: {sizes_sorted[:5]}")
print(f"{'='*45}")
