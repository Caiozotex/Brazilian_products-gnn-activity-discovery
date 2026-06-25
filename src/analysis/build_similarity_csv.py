"""
Builds a CSV of all compound pairs with cosine similarity >= THRESHOLD.

Columns:
    joint_idx_i, joint_idx_j, similarity

Run AFTER count_similar_pairs.py to pick a threshold you're happy with.

Usage:
    python -m src.analysis.build_similarity_csv --threshold 0.8

Output:
    results/embeddings/similar_pairs_<threshold>.csv
"""

import argparse
import numpy as np
import pandas as pd
import os

EMB_PATH  = "results/embeddings/joint_embeddings.npy"
META_PATH = "results/embeddings/joint_embeddings_meta.csv"
OUT_DIR   = "results/embeddings"
CHUNK     = 512


def main(threshold: float):
    print(f"Loading embeddings ...")
    emb  = np.load(EMB_PATH).astype(np.float32)
    meta = pd.read_csv(META_PATH)
    N    = emb.shape[0]

    # joint_idx for each row in the embeddings array
    joint_ids = meta["joint_idx"].values   # shape (N,)

    # L2-normalize for cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb   = emb / norms
    print(f"  {N} compounds normalized")
    print(f"  threshold = {threshold}\n")

    out_path = os.path.join(OUT_DIR, f"similar_pairs_{threshold:.2f}.csv")

    # Stream results directly to CSV to avoid holding everything in RAM
    first_write = True
    total_pairs_written = 0

    n_chunks = (N + CHUNK - 1) // CHUNK
    print(f"Processing {n_chunks} chunks ...")

    for chunk_start in range(0, N, CHUNK):
        chunk_end  = min(chunk_start + CHUNK, N)
        chunk      = emb[chunk_start:chunk_end]          # (B, 128)
        sim_block  = chunk @ emb.T                       # (B, N)

        rows_i = []
        rows_j = []
        rows_s = []

        for local_i, global_i in enumerate(range(chunk_start, chunk_end)):
            # upper triangle only: j > i
            sim_row = sim_block[local_i, global_i + 1:]
            js      = np.where(sim_row >= threshold)[0] + global_i + 1

            if len(js) == 0:
                continue

            rows_i.extend([joint_ids[global_i]] * len(js))
            rows_j.extend(joint_ids[js].tolist())
            rows_s.extend(sim_row[js - global_i - 1].tolist())

        if rows_i:
            df_chunk = pd.DataFrame({
                "joint_idx_i": rows_i,
                "joint_idx_j": rows_j,
                "similarity":  rows_s,
            })
            df_chunk["similarity"] = df_chunk["similarity"].round(6)
            df_chunk.to_csv(
                out_path,
                mode="a",
                index=False,
                header=first_write,
            )
            first_write = False
            total_pairs_written += len(df_chunk)

        if (chunk_start // CHUNK) % 10 == 0:
            pct = chunk_end / N * 100
            print(f"  {chunk_end}/{N} ({pct:.1f}%)  pairs so far: {total_pairs_written:,}")

    print(f"\nDone. {total_pairs_written:,} pairs written to:")
    print(f"  {out_path}")
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  file size: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold", type=float, default=0.8,
        help="Minimum cosine similarity to include a pair (default: 0.8)"
    )
    args = parser.parse_args()
    main(args.threshold)
