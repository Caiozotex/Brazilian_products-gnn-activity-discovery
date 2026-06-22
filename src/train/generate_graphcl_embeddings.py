"""
Generate graph-level embeddings using the GraphCL-pretrained encoder.

Reads graphs from the joint HIV + BrNPDB dataset (via joint_embeddings_meta.csv)
and saves:
  results/embeddings/graphcl_embeddings.npy        (N, 128)
  results/embeddings/graphcl_embeddings_meta.csv   (joint_idx, source, ...)
"""

import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from src.models.gin_model import GINEEncoder, GraphPooling

META_CSV         = "results/embeddings/joint_embeddings_meta.csv"
HIV_GRAPH_DIR    = "data/processed/graphs_hiv"
BRNPDB_GRAPH_DIR = "data/processed/graphs_brnpdb_antiviral"
ENCODER_CKPT     = "models/checkpoints/graphcl/best_encoder.pt"
EMB_DIR          = "results/embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# Load encoder
# ──────────────────────────────────────────────────────────────
print(f"Loading GraphCL encoder from {ENCODER_CKPT} ...")
encoder = GINEEncoder(in_channels=35, edge_dim=9, hidden_channels=128, num_layers=4, dropout=0.2)
encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=DEVICE, weights_only=True))
encoder.to(DEVICE)
encoder.eval()
pool = GraphPooling()
print(f"  Device: {DEVICE}")

# ──────────────────────────────────────────────────────────────
# Load graphs in joint_idx order
# ──────────────────────────────────────────────────────────────
print(f"\nLoading graphs from {META_CSV} ...")
meta = pd.read_csv(META_CSV)

graphs   = []
kept_idx = []

for _, row in meta.iterrows():
    source    = row["source"]
    jidx      = int(row["joint_idx"])
    brnpdb_id = row.get("brnpdb_id", None)

    if source == "HIV":
        path = os.path.join(HIV_GRAPH_DIR, f"{jidx}.pt")
    else:
        path = os.path.join(BRNPDB_GRAPH_DIR, f"{brnpdb_id}.pt")

    if not os.path.exists(path):
        continue

    data = torch.load(path, map_location="cpu", weights_only=False)
    graphs.append(data)
    kept_idx.append(jidx)

print(f"  {len(graphs)} graphs loaded  ({len(meta) - len(graphs)} missing)")

# ──────────────────────────────────────────────────────────────
# Generate embeddings in batches
# ──────────────────────────────────────────────────────────────
print("\nGenerating embeddings ...")

@torch.no_grad()
def embed_graphs(graph_list, batch_size=256):
    loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False)
    all_emb = []
    for batch in loader:
        batch     = batch.to(DEVICE)
        node_emb  = encoder(batch.x, batch.edge_index, batch.edge_attr)
        graph_emb = pool(node_emb, batch.batch)
        all_emb.append(graph_emb.cpu())
    return torch.cat(all_emb, dim=0).numpy()

emb = embed_graphs(graphs)
print(f"  shape: {emb.shape}")

# ──────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────
emb_path  = os.path.join(EMB_DIR, "graphcl_embeddings.npy")
meta_path = os.path.join(EMB_DIR, "graphcl_embeddings_meta.csv")

np.save(emb_path, emb)

meta_out = meta.set_index("joint_idx").loc[kept_idx].copy()
meta_out.index.name = "joint_idx"
meta_out = meta_out.reset_index()
meta_out["emb_row"] = range(len(meta_out))
meta_out.to_csv(meta_path, index=False)

print(f"\nSaved embeddings -> {emb_path}  shape: {emb.shape}")
print(f"Saved metadata   -> {meta_path}  ({len(meta_out)} rows)")
print("\nDone.")
