import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

CHUNK = 512
EMB_PATH  = "results/graphcl/graphcl_embeddings.npy"
META_PATH = "results/graphcl/graphcl_embeddings_meta.csv"
JOINT_CSV = "data/processed/joint_data.csv"


def count_edges_components(sim_matrix_fn, N, thresholds, label):
    print(f"\n=== {label} ===")
    print(f"{'Threshold':>10}  {'Edges':>12}  {'Components':>12}  {'Largest':>10}  {'Isolated':>10}")
    print("-" * 62)
    for thresh in thresholds:
        rows_i, rows_j = [], []
        for s in range(0, N, CHUNK):
            e = min(s + CHUNK, N)
            sim = sim_matrix_fn(s, e)
            for li, gi in enumerate(range(s, e)):
                row = sim[li, gi + 1:]
                js  = np.where(row >= thresh)[0] + gi + 1
                if len(js):
                    rows_i.extend([gi] * len(js))
                    rows_j.extend(js.tolist())

        n_edges = len(rows_i)
        if n_edges == 0:
            print(f"{thresh:>10.2f}  {0:>12,}  {N:>12,}  {1:>10,}  {N:>10,}")
            continue

        ri = np.array(rows_i, dtype=np.int32)
        rj = np.array(rows_j, dtype=np.int32)
        data = np.ones(n_edges * 2, dtype=np.bool_)
        r    = np.concatenate([ri, rj])
        c    = np.concatenate([rj, ri])
        adj  = sp.csr_matrix((data, (r, c)), shape=(N, N), dtype=np.bool_)
        n_comp, labels = csgraph.connected_components(adj, directed=False)
        sizes    = np.bincount(labels)
        largest  = sizes.max()
        isolated = (sizes == 1).sum()
        print(f"{thresh:>10.2f}  {n_edges:>12,}  {n_comp:>12,}  {largest:>10,}  {isolated:>10,}")


# ── Load embeddings ────────────────────────────────────────────
emb_raw = np.load(EMB_PATH).astype(np.float32)
N = emb_raw.shape[0]
print(f"N = {N}")

# ── 1. Euclidean similarity: sim = 1 / (1 + ||a - b||) ───────
# Computed via ||a-b||^2 = ||a||^2 + ||b||^2 - 2*(a·b)
sq = (emb_raw ** 2).sum(axis=1)

def euclidean_sim(s, e):
    chunk = emb_raw[s:e]
    dot   = chunk @ emb_raw.T
    dist2 = sq[s:e, None] + sq[None, :] - 2 * dot
    dist2 = np.maximum(dist2, 0.0)          # numerical safety
    return 1.0 / (1.0 + np.sqrt(dist2))

count_edges_components(euclidean_sim, N,
                       thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
                       label="Euclidean sim = 1/(1+dist)  (GraphCL embeddings)")

# ── 2. Tanimoto on Morgan fingerprints ────────────────────────
print("\nComputing Morgan fingerprints ...")
import warnings; warnings.filterwarnings("ignore")
meta  = pd.read_csv(META_PATH)
joint = pd.read_csv(JOINT_CSV)
idx_to_smiles = joint.set_index("joint_idx")["smiles"].to_dict()

fps = []
for jidx in meta["joint_idx"]:
    smi = idx_to_smiles.get(jidx, None)
    mol = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) else None
    if mol is None:
        fps.append(np.zeros(2048, dtype=np.float32))
        continue
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    fps.append(arr)

fps_mat = np.stack(fps)
cnt_all = fps_mat.sum(axis=1, keepdims=True)
print("Fingerprints ready.")

def tanimoto_sim(s, e):
    chunk = fps_mat[s:e]
    inter = chunk @ fps_mat.T
    union = cnt_all[s:e] + cnt_all.T - inter
    return np.where(union > 0, inter / union, 0.0)

count_edges_components(tanimoto_sim, N,
                       thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
                       label="Tanimoto (Morgan fp, radius=2, 2048 bits)")
