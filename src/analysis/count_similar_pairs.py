"""
Counts compound pairs above similarity thresholds.

Methods:
  cosine          -- cosine on raw GNN embeddings
  centered_cosine -- cosine after subtracting global mean
  tanimoto        -- Tanimoto on Morgan fingerprints (radius=2, 2048 bits)
  hybrid          -- alpha*tanimoto + (1-alpha)*centered_cosine  [recommended]

Usage:
    python -m src.analysis.count_similar_pairs --method hybrid --threshold 0.5
    python -m src.analysis.count_similar_pairs --method tanimoto --threshold 0.7
"""

import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
import warnings
warnings.filterwarnings("ignore")

EMB_PATH   = "results/embeddings/joint_embeddings.npy"
META_PATH  = "results/embeddings/joint_embeddings_meta.csv"
JOINT_CSV  = "data/processed/joint_data.csv"
CHUNK      = 256   # smaller chunk: hybrid loads both matrices simultaneously

# Thresholds explored in the table (always includes user threshold)
DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Alpha values explored when method=hybrid
ALPHAS = [0.3, 0.5, 0.7, 0.9]


# ──────────────────────────────────────────────────────────────
# Feature builders
# ──────────────────────────────────────────────────────────────

def load_centered_cosine():
    emb = np.load(EMB_PATH).astype(np.float32)
    emb = emb - emb.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norms == 0, 1.0, norms)
    return emb   # (N, 128), unit vectors


def load_raw_embeddings():
    emb = np.load(EMB_PATH).astype(np.float32)
    return emb   # (N, 128), raw (not normalized)


def load_morgan_fps(radius=2, nbits=2048):
    joint = pd.read_csv(JOINT_CSV)
    meta  = pd.read_csv(META_PATH)
    idx_to_smiles = joint.set_index("joint_idx")["smiles"].to_dict()

    fps, valid = [], []
    for jidx in meta["joint_idx"]:
        smi = idx_to_smiles.get(jidx, None)
        mol = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) else None
        if mol is None:
            fps.append(None); valid.append(False); continue
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros(nbits, dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        fps.append(arr); valid.append(True)

    valid_idx  = [i for i, v in enumerate(valid) if v]
    fp_matrix  = np.stack([fps[i] for i in valid_idx])   # (N_valid, nbits)
    n_invalid  = sum(not v for v in valid)
    if n_invalid:
        print(f"  ! {n_invalid} invalid SMILES excluded from fingerprints")
    return fp_matrix, valid_idx


# ──────────────────────────────────────────────────────────────
# Per-block similarity computations
# ──────────────────────────────────────────────────────────────

def cosine_block(chunk, matrix):
    return chunk @ matrix.T   # inner product on unit vectors = cosine sim


def euclidean_sim_block(chunk, matrix):
    """sim = 1 / (1 + L2_distance), ranges in (0, 1]."""
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    dot       = chunk @ matrix.T                                        # (B, N)
    norm_c_sq = (chunk  ** 2).sum(axis=1, keepdims=True)               # (B, 1)
    norm_m_sq = (matrix ** 2).sum(axis=1, keepdims=True).T             # (1, N)
    dist      = np.sqrt(np.clip(norm_c_sq + norm_m_sq - 2 * dot, 0, None))
    return 1.0 / (1.0 + dist)


def tanimoto_block(chunk, matrix):
    intersection  = chunk @ matrix.T
    count_chunk   = chunk.sum(axis=1, keepdims=True)
    count_matrix  = matrix.sum(axis=1, keepdims=True).T
    union         = count_chunk + count_matrix - intersection
    return np.where(union > 0, intersection / union, 0.0)


def hybrid_block(cc_chunk, cc_matrix, tan_chunk, tan_matrix, alpha):
    t = tanimoto_block(tan_chunk, tan_matrix)
    c = cosine_block(cc_chunk, cc_matrix)
    return alpha * t + (1.0 - alpha) * c


# ──────────────────────────────────────────────────────────────
# Generic counter
# ──────────────────────────────────────────────────────────────

def count_pairs(N, thresholds, get_sim_row_fn):
    """
    Iterates over upper-triangle pairs in chunks.
    get_sim_row_fn(global_i) -> 1-D array of similarities for j > global_i
    """
    counts = {t: 0 for t in thresholds}
    n_chunks = (N + CHUNK - 1) // CHUNK

    for chunk_start in range(0, N, CHUNK):
        chunk_end = min(chunk_start + CHUNK, N)

        sim_block = get_sim_row_fn(chunk_start, chunk_end)   # (B, N)

        for local_i, global_i in enumerate(range(chunk_start, chunk_end)):
            sim_row = sim_block[local_i, global_i + 1:]
            for t in thresholds:
                counts[t] += int((sim_row >= t).sum())

        if (chunk_start // CHUNK) % 10 == 0:
            pct = chunk_end / N * 100
            print(f"  {chunk_end}/{N} ({pct:.1f}%)")

    return counts


# ──────────────────────────────────────────────────────────────
# Pretty-print table
# ──────────────────────────────────────────────────────────────

def print_table(counts, total_pairs, user_threshold, label=""):
    print(f"\n{'=' * 58}")
    if label:
        print(f"  {label}")
    print(f"{'Threshold':>10}  {'Pairs':>15}  {'% of total':>12}")
    print(f"{'-' * 58}")
    for t in sorted(counts):
        c   = counts[t]
        pct = c / total_pairs * 100
        marker = "  <--" if t == user_threshold else ""
        print(f"{t:>10.2f}  {c:>15,}  {pct:>11.4f}%{marker}")
    print(f"{'=' * 58}")
    c   = counts[user_threshold]
    est = c * 30 / 1e6
    print(f"  At threshold={user_threshold}: {c:,} pairs  (~{est:.1f} MB CSV)\n")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(method: str, user_threshold: float, alphas, extra_thresholds):
    thresholds = sorted(set(DEFAULT_THRESHOLDS + [user_threshold] + extra_thresholds))

    if method == "euclidean":
        print("Loading raw embeddings (no normalization) ...")
        emb = load_raw_embeddings()
        N   = emb.shape[0]
        total_pairs = N * (N - 1) // 2

        # Print distance stats to calibrate thresholds
        sample_idx = np.random.choice(N, min(500, N), replace=False)
        sample     = emb[sample_idx]
        sample_sim = euclidean_sim_block(sample, sample)
        upper      = sample_sim[np.triu_indices(len(sample), k=1)]
        print(f"  L2 similarity (1/(1+d)) sample stats:")
        print(f"    mean={upper.mean():.4f}  median={np.median(upper):.4f}  "
              f"p10={np.percentile(upper,10):.4f}  p90={np.percentile(upper,90):.4f}")
        print(f"Method: euclidean | N={N:,} | pairs={total_pairs:,}\n")

        def get_block(s, e):
            return euclidean_sim_block(emb[s:e], emb)

        counts = count_pairs(N, thresholds, get_block)
        print_table(counts, total_pairs, user_threshold, "euclidean (sim = 1/(1+L2))")

    elif method in ("cosine", "centered_cosine"):
        emb = load_centered_cosine() if method == "centered_cosine" else (
            lambda: (lambda e: e / np.where(
                (n := np.linalg.norm(e, axis=1, keepdims=True)) == 0, 1.0, n
            ))(np.load(EMB_PATH).astype(np.float32))
        )()
        N = emb.shape[0]
        total_pairs = N * (N - 1) // 2
        print(f"Method: {method} | N={N:,} | pairs C(N,2)={total_pairs:,}\n")

        def get_block(s, e):
            return cosine_block(emb[s:e], emb)

        counts = count_pairs(N, thresholds, get_block)
        print_table(counts, total_pairs, user_threshold, method)

    elif method == "tanimoto":
        print("Loading Morgan fingerprints ...")
        fps, _ = load_morgan_fps()
        N = fps.shape[0]
        total_pairs = N * (N - 1) // 2
        print(f"Method: tanimoto | N={N:,} | pairs C(N,2)={total_pairs:,}\n")

        def get_block(s, e):
            return tanimoto_block(fps[s:e], fps)

        counts = count_pairs(N, thresholds, get_block)
        print_table(counts, total_pairs, user_threshold, "tanimoto (Morgan r=2, 2048 bits)")

    elif method == "hybrid":
        print("Loading centered cosine embeddings ...")
        cc = load_centered_cosine()
        print("Loading Morgan fingerprints ...")
        fps, valid_idx = load_morgan_fps()

        # restrict to rows present in both (fingerprints may exclude invalid SMILES)
        valid_set = set(valid_idx)
        keep = [i for i in range(len(cc)) if i in valid_set]
        cc_mat  = cc[keep]
        fps_mat = fps   # already only valid rows, in same order as keep

        N = len(keep)
        total_pairs = N * (N - 1) // 2
        print(f"Method: hybrid | N={N:,} | pairs C(N,2)={total_pairs:,}")
        print(f"Alphas to explore: {alphas}\n")

        for alpha in alphas:
            print(f"--- alpha={alpha}  (hybrid = {alpha}*tanimoto + {1-alpha:.1f}*centered_cosine) ---")

            def get_block(s, e, a=alpha):
                return hybrid_block(cc_mat[s:e], cc_mat,
                                    fps_mat[s:e], fps_mat, a)

            counts = count_pairs(N, thresholds, get_block)
            print_table(counts, total_pairs, user_threshold,
                        f"hybrid alpha={alpha}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["cosine", "centered_cosine", "tanimoto", "hybrid", "euclidean"],
                        default="euclidean")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS,
                        help="Alpha values to explore (hybrid only)")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[],
                        help="Extra threshold values to include in the table")
    args = parser.parse_args()
    main(args.method, args.threshold, args.alphas, args.thresholds)
