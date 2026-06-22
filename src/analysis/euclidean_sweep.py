import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

emb_raw = np.load("results/graphcl/graphcl_embeddings.npy").astype(np.float32)
N   = emb_raw.shape[0]
sq  = (emb_raw ** 2).sum(axis=1)
CHUNK = 512

# ── Sample pairwise distances to choose thresholds ────────────
rng = np.random.default_rng(42)
idx = rng.choice(N, size=2000, replace=False)
sub = emb_raw[idx]
sq_sub = (sub ** 2).sum(axis=1)
dot = sub @ sub.T
d2  = sq_sub[:, None] + sq_sub[None, :] - 2.0 * dot
d   = np.sqrt(np.maximum(d2, 0.0))
# upper triangle only
mask = np.triu(np.ones_like(d, dtype=bool), k=1)
d_sample = d[mask]
sim_sample = 1.0 / (1.0 + d_sample)

print(f"N = {N}")
print(f"\nPairwise distance sample (n=~2M pairs):")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    dv = np.percentile(d_sample, p)
    sv = np.percentile(sim_sample, 100 - p)
    print(f"  p{p:>2}: dist={dv:.3f}  sim=1/(1+d)={sv:.4f}")

# Pick thresholds from high-sim end
thresholds = sorted(set(
    [round(np.percentile(sim_sample, p), 3) for p in [80, 85, 90, 95, 99]]
), reverse=True)
print(f"\nAuto-selected thresholds (high-sim percentiles): {thresholds}")

# ── Sweep ─────────────────────────────────────────────────────
print(f"\nMetric: Euclidean sim = 1 / (1 + ||a-b||)  (GraphCL embeddings)")
print(f"{'Threshold':>10}  {'Edges':>12}  {'Components':>12}  {'Largest':>10}  {'Isolated':>10}")
print("-" * 65)

for thresh in thresholds:
    ri, rj = [], []
    for s in range(0, N, CHUNK):
        e   = min(s + CHUNK, N)
        dot = emb_raw[s:e] @ emb_raw.T
        d2  = sq[s:e, None] + sq[None, :] - 2.0 * dot
        sim = 1.0 / (1.0 + np.sqrt(np.maximum(d2, 0.0)))
        for li, gi in enumerate(range(s, e)):
            row = sim[li, gi + 1:]
            js  = np.where(row >= thresh)[0] + gi + 1
            if len(js):
                ri.extend([gi] * len(js))
                rj.extend(js.tolist())

    n = len(ri)
    if n == 0:
        print(f"{thresh:>10.4f}  {0:>12,}  {N:>12,}  {1:>10,}  {N:>10,}")
        continue

    data = np.ones(n * 2, dtype=np.int8)
    r    = np.concatenate([np.array(ri, dtype=np.int32), np.array(rj, dtype=np.int32)])
    c    = np.concatenate([np.array(rj, dtype=np.int32), np.array(ri, dtype=np.int32)])
    adj  = sp.csr_matrix((data, (r, c)), shape=(N, N), dtype=np.int8)
    nc, lab = csgraph.connected_components(adj, directed=False)
    sz = np.bincount(lab)
    print(f"{thresh:>10.4f}  {n:>12,}  {nc:>12,}  {sz.max():>10,}  {(sz==1).sum():>10,}")
