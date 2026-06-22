"""
Generate joint_data.csv and graph-level embeddings for HIV + BrNPDB antiviral compounds.

Steps:
1. Build joint_data.csv: HIV.csv rows followed by brnpdb_id.csv rows, with a
   unified integer index (joint_idx) and a 'source' column.
2. Load the trained HIV encoder checkpoint.
3. For each row in joint_data.csv that has a processed graph (.pt file), run the
   GINEEncoder + mean pooling to get a 128-dim embedding.
4. Save:
   - data/external/joint_data.csv
   - results/embeddings/joint_embeddings.npy  (shape: total_valid, 128)
   - results/embeddings/joint_embeddings_meta.csv  (joint_idx, source, smiles, ...)
"""

import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader

from src.models.gin_model import GINEEncoder, GraphPooling

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
HIV_CSV        = "data/external/HIV.csv"
BRNPDB_CSV     = "data/external/brnpdb_id.csv"
JOINT_CSV      = "data/external/joint_data.csv"

HIV_GRAPH_DIR     = "data/processed/graphs_hiv"
BRNPDB_GRAPH_DIR  = "data/processed/graphs_brnpdb_antiviral"

ENCODER_CKPT   = "models/checkpoints/hiv/best_encoder.pt"

EMB_DIR        = "results/embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# 1. Build joint_data.csv
# ──────────────────────────────────────────────────────────────
print("Building joint_data.csv ...")

hiv_df = pd.read_csv(HIV_CSV)
hiv_df["source"]      = "HIV"
hiv_df["brnpdb_id"]   = pd.NA
hiv_df["common_name"] = pd.NA
# 'activity' column already exists in HIV (CI/CM/CA)

brnpdb_df = pd.read_csv(BRNPDB_CSV)
# brnpdb_id.csv columns: index, brnpdb_id, common_name, smiles
brnpdb_df["source"]     = "brnpdb"
brnpdb_df["HIV_active"] = pd.NA   # unknown HIV activity
brnpdb_df["activity"]   = pd.NA   # not from the HIV assay

# align columns
COLS = ["smiles", "activity", "HIV_active", "brnpdb_id", "common_name", "source"]
joint = pd.concat(
    [hiv_df[COLS], brnpdb_df[COLS]],
    ignore_index=True
)
joint.index.name = "joint_idx"
joint = joint.reset_index()          # joint_idx becomes an explicit column

joint.to_csv(JOINT_CSV, index=False)
print(f"  saved: {JOINT_CSV}")
print(f"  total rows: {len(joint)}  ({len(hiv_df)} HIV + {len(brnpdb_df)} BrNPDB)")

# ──────────────────────────────────────────────────────────────
# 2. Load encoder
# ──────────────────────────────────────────────────────────────
print("\nLoading encoder ...")
encoder = GINEEncoder(
    in_channels=35,
    edge_dim=9,
    hidden_channels=128,
    num_layers=4,
    dropout=0.2,
)
encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=DEVICE, weights_only=True))
encoder.to(DEVICE)
encoder.eval()
pool = GraphPooling()
print(f"  encoder loaded from {ENCODER_CKPT}")

# ──────────────────────────────────────────────────────────────
# 3. Helper: embed a list of .pt graphs in batches
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def embed_graphs(graph_list, batch_size=256):
    loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False)
    all_emb = []
    for batch in loader:
        batch = batch.to(DEVICE)
        node_emb = encoder(batch.x, batch.edge_index, batch.edge_attr)
        graph_emb = pool(node_emb, batch.batch)
        all_emb.append(graph_emb.cpu())
    return torch.cat(all_emb, dim=0).numpy()

# ──────────────────────────────────────────────────────────────
# 4. Load HIV graphs (in joint_idx order, skip missing)
# ──────────────────────────────────────────────────────────────
print("\nLoading HIV graphs ...")
hiv_graphs   = []
hiv_meta_idx = []   # joint_idx values that have a valid graph

hiv_rows = joint[joint["source"] == "HIV"]
for _, row in hiv_rows.iterrows():
    # Graph files are named by original HIV.csv row index which equals joint_idx
    original_hiv_idx = row["joint_idx"]
    pt_path = os.path.join(HIV_GRAPH_DIR, f"{original_hiv_idx}.pt")
    if not os.path.exists(pt_path):
        continue
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    hiv_graphs.append(data)
    hiv_meta_idx.append(int(row["joint_idx"]))

print(f"  {len(hiv_graphs)} valid HIV graphs  ({len(hiv_rows) - len(hiv_graphs)} skipped / no .pt file)")

# ──────────────────────────────────────────────────────────────
# 5. Load BrNPDB graphs (in joint_idx order)
# ──────────────────────────────────────────────────────────────
print("Loading BrNPDB antiviral graphs ...")
brnpdb_graphs   = []
brnpdb_meta_idx = []

brnpdb_rows = joint[joint["source"] == "brnpdb"]
for _, row in brnpdb_rows.iterrows():
    bid = row["brnpdb_id"]
    pt_path = os.path.join(BRNPDB_GRAPH_DIR, f"{bid}.pt")
    if not os.path.exists(pt_path):
        print(f"  ! Missing graph for brnpdb_id={bid}, skipping")
        continue
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    brnpdb_graphs.append(data)
    brnpdb_meta_idx.append(int(row["joint_idx"]))

print(f"  {len(brnpdb_graphs)} valid BrNPDB graphs")

# ──────────────────────────────────────────────────────────────
# 6. Generate embeddings
# ──────────────────────────────────────────────────────────────
print("\nGenerating HIV embeddings ...")
hiv_emb = embed_graphs(hiv_graphs)
print(f"  shape: {hiv_emb.shape}")

print("Generating BrNPDB embeddings ...")
brnpdb_emb = embed_graphs(brnpdb_graphs)
print(f"  shape: {brnpdb_emb.shape}")

# ──────────────────────────────────────────────────────────────
# 7. Concatenate and save
# ──────────────────────────────────────────────────────────────
all_emb = np.vstack([hiv_emb, brnpdb_emb])
all_idx = hiv_meta_idx + brnpdb_meta_idx

emb_path = os.path.join(EMB_DIR, "joint_embeddings.npy")
np.save(emb_path, all_emb)
print(f"\nSaved embeddings -> {emb_path}  shape: {all_emb.shape}")

# Build metadata CSV aligned with the embeddings array
meta_rows = joint.set_index("joint_idx").loc[all_idx].copy()
meta_rows.index.name = "joint_idx"
meta_rows = meta_rows.reset_index()
meta_rows["emb_row"] = range(len(meta_rows))   # row index into joint_embeddings.npy

meta_path = os.path.join(EMB_DIR, "joint_embeddings_meta.csv")
meta_rows.to_csv(meta_path, index=False)
print(f"Saved metadata   -> {meta_path}  ({len(meta_rows)} rows)")

print("\nDone.")
print(f"  joint_data.csv  : {len(joint)} total rows")
print(f"  embeddings      : {all_emb.shape[0]} compounds x {all_emb.shape[1]} dims")
print(f"    HIV           : {len(hiv_graphs)}")
print(f"    BrNPDB        : {len(brnpdb_graphs)}")
