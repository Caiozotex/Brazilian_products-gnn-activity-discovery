import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

CHUNK     = 512
JOINT_CSV = "data/processed/joint_data.csv"
EMB_PATH  = "results/embeddings/joint_embeddings.npy"
META_PATH = "results/embeddings/joint_embeddings_meta.csv"

# ── Load embeddings ────────────────────────────────────────────
emb = np.load(EMB_PATH).astype(np.float32)
N   = emb.shape[0]
print(f"Old HIV GNN embeddings: N={N:,}, dim={emb.shape[1]}")

# Centered cosine
cc = emb - emb.mean(axis=0, keepdims=True)
cc /= np.where(np.linalg.norm(cc, axis=1, keepdims=True) == 0, 1.0,
               np.linalg.norm(cc, axis=1, keepdims=True))

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
print("Done.\n")

# ── Helper ─────────────────────────────────────────────────────
def stats(sim_fn, thresh):
    # Accumulate as numpy arrays to avoid Python list overhead
    ri_parts, rj_parts = [], []
    n_total = 0
    for s in range(0, N, CHUNK):
        e   = min(s + CHUNK, N)
        sim = sim_fn(s, e)
        for li, gi in enumerate(range(s, e)):
            row = sim[li, gi + 1:]
            js  = (np.where(row >= thresh)[0] + gi + 1).astype(np.int32)
            if len(js):
                ri_parts.append(np.full(len(js), gi, dtype=np.int32))
                rj_parts.append(js)
                n_total += len(js)
    if n_total == 0:
        return 0, N, 1, N
    ri = np.concatenate(ri_parts)
    rj = np.concatenate(rj_parts)
    data = np.ones(n_total * 2, dtype=np.int8)
    r = np.concatenate([ri, rj])
    c = np.concatenate([rj, ri])
    adj = sp.csr_matrix((data, (r, c)), shape=(N, N), dtype=np.int8)
    nc, lab = csgraph.connected_components(adj, directed=False)
    sz = np.bincount(lab)
    return n_total, nc, sz.max(), (sz == 1).sum()

def header(title):
    print(f"\n--- {title} ---")
    print(f"  {'thresh':>6}  {'edges':>10}  {'components':>12}  {'largest':>9}  {'isolated':>9}")
    print(f"  {'-'*54}")

def row(thresh, n, nc, largest, iso):
    print(f"  {thresh:>6.2f}  {n:>10,}  {nc:>12,}  {largest:>9,}  {iso:>9,}")

# ── Centered cosine ────────────────────────────────────────────
header("Centered cosine  (old GNN)")
for t in [0.84, 0.90, 0.95]:
    row(t, *stats(lambda s, e: cc[s:e] @ cc.T, t))

# ── Tanimoto ───────────────────────────────────────────────────
def tan(s, e):
    inter = fps_mat[s:e] @ fps_mat.T
    union = cnt_all[s:e] + cnt_all.T - inter
    return np.where(union > 0, inter / union, 0.0)

header("Tanimoto Morgan fp  (old GNN)")
for t in [0.30, 0.40, 0.50, 0.60]:
    row(t, *stats(tan, t))

# ── Hybrid alpha=0.2 ───────────────────────────────────────────
def hyb(s, e):
    inter = fps_mat[s:e] @ fps_mat.T
    union = cnt_all[s:e] + cnt_all.T - inter
    t = np.where(union > 0, inter / union, 0.0)
    return 0.2 * t + 0.8 * (cc[s:e] @ cc.T)

header("Hybrid alpha=0.2  (old GNN)")
for t in [0.80, 0.84, 0.90]:
    row(t, *stats(hyb, t))
