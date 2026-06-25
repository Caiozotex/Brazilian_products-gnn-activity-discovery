"""
Compare similarity methods on old HIV GNN embeddings vs GraphCL embeddings.
Metrics: centered cosine, pure cosine, Tanimoto, hybrid (alpha=0.2)
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

CHUNK = 512
JOINT_CSV   = "data/processed/joint_data.csv"
OLD_EMB     = "results/embeddings/joint_embeddings.npy"
OLD_META    = "results/embeddings/joint_embeddings_meta.csv"
GCL_EMB     = "results/graphcl/graphcl_embeddings.npy"
GCL_META    = "results/graphcl/graphcl_embeddings_meta.csv"


def report(label, thresh, n_edges, N, adj):
    nc, lab = csgraph.connected_components(adj, directed=False)
    sz = np.bincount(lab)
    print(f"  {label:<30} thresh={thresh:.2f}  edges={n_edges:>9,}  "
          f"components={nc:>6,}  largest={sz.max():>6,}  isolated={( sz==1).sum():>6,}")


def sweep(emb_raw, fps_mat, label_prefix, thresholds_cos, thresholds_tan, thresholds_hybrid):
    N = emb_raw.shape[0]

    # Centered cosine
    cc = emb_raw - emb_raw.mean(axis=0, keepdims=True)
    cc /= np.where(np.linalg.norm(cc, axis=1, keepdims=True) == 0, 1.0,
                   np.linalg.norm(cc, axis=1, keepdims=True))

    # Pure cosine
    pc = emb_raw / np.where(np.linalg.norm(emb_raw, axis=1, keepdims=True) == 0, 1.0,
                             np.linalg.norm(emb_raw, axis=1, keepdims=True))

    cnt_all = fps_mat.sum(axis=1, keepdims=True)

    def build_adj(sim_fn, thresh):
        ri, rj = [], []
        for s in range(0, N, CHUNK):
            e   = min(s + CHUNK, N)
            sim = sim_fn(s, e)
            for li, gi in enumerate(range(s, e)):
                row = sim[li, gi + 1:]
                js  = np.where(row >= thresh)[0] + gi + 1
                if len(js):
                    ri.extend([gi] * len(js)); rj.extend(js.tolist())
        n = len(ri)
        if n == 0:
            return 0, sp.csr_matrix((N, N), dtype=np.int8)
        data = np.ones(n * 2, dtype=np.int8)
        r = np.concatenate([np.array(ri, np.int32), np.array(rj, np.int32)])
        c = np.concatenate([np.array(rj, np.int32), np.array(ri, np.int32)])
        return n, sp.csr_matrix((data, (r, c)), shape=(N, N), dtype=np.int8)

    print(f"\n{'='*75}")
    print(f"  {label_prefix}  (N={N:,})")
    print(f"{'='*75}")

    # Centered cosine
    for t in thresholds_cos:
        n, adj = build_adj(lambda s, e: cc[s:e] @ cc.T, t)
        report("Centered cosine", t, n, N, adj)

    # Pure cosine
    for t in thresholds_cos:
        n, adj = build_adj(lambda s, e: pc[s:e] @ pc.T, t)
        report("Pure cosine", t, n, N, adj)

    # Tanimoto
    def tan_fn(s, e):
        chunk = fps_mat[s:e]
        inter = chunk @ fps_mat.T
        union = cnt_all[s:e] + cnt_all.T - inter
        return np.where(union > 0, inter / union, 0.0)

    for t in thresholds_tan:
        n, adj = build_adj(tan_fn, t)
        report("Tanimoto", t, n, N, adj)

    # Hybrid alpha=0.2
    def hyb_fn(s, e):
        chunk_tan = fps_mat[s:e]
        inter = chunk_tan @ fps_mat.T
        union = cnt_all[s:e] + cnt_all.T - inter
        tan = np.where(union > 0, inter / union, 0.0)
        cos = cc[s:e] @ cc.T
        return 0.2 * tan + 0.8 * cos

    for t in thresholds_hybrid:
        n, adj = build_adj(hyb_fn, t)
        report("Hybrid (alpha=0.2)", t, n, N, adj)


# ── Load Morgan fingerprints (same for both) ──────────────────
print("Loading Morgan fingerprints ...")
joint = pd.read_csv(JOINT_CSV)
meta  = pd.read_csv(OLD_META)
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
fps_old = np.stack(fps)
print(f"  {len(fps_old)} fingerprints (old meta, {len(meta)} rows)")

# For GraphCL meta (may have different order / size)
meta_gcl = pd.read_csv(GCL_META)
fps_gcl = []
for jidx in meta_gcl["joint_idx"]:
    smi = idx_to_smiles.get(jidx, None)
    mol = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) else None
    if mol is None:
        fps_gcl.append(np.zeros(2048, dtype=np.float32)); continue
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    ConvertToNumpyArray(fp, arr); fps_gcl.append(arr)
fps_gcl = np.stack(fps_gcl)
print(f"  {len(fps_gcl)} fingerprints (graphcl meta)")

# ── Run sweeps ────────────────────────────────────────────────
COS_THRESHOLDS    = [0.80, 0.84, 0.90]
TAN_THRESHOLDS    = [0.30, 0.40, 0.50]
HYBRID_THRESHOLDS = [0.80, 0.84, 0.90]

emb_old = np.load(OLD_EMB).astype(np.float32)
sweep(emb_old, fps_old, "OLD HIV GNN encoder", COS_THRESHOLDS, TAN_THRESHOLDS, HYBRID_THRESHOLDS)

emb_gcl = np.load(GCL_EMB).astype(np.float32)
sweep(emb_gcl, fps_gcl, "GraphCL encoder", COS_THRESHOLDS, TAN_THRESHOLDS, HYBRID_THRESHOLDS)
