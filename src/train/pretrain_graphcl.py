"""
GraphCL contrastive pretraining on the joint HIV + BrNPDB graph dataset.

For each molecule, two structurally-augmented views are generated and the
NT-Xent loss pulls their embeddings together while pushing all other
molecules apart. No labels are used — the encoder learns molecular structure.

Augmentations applied independently to each view:
  1. drop_edges       — randomly remove bonds (p=0.15)
  2. mask_node_feats  — randomly zero atom features (p=0.15)
  (randomly chooses one augmentation per view to keep diversity)

Checkpoint saved to:
  models/checkpoints/graphcl/best_encoder.pt   <- use this downstream
  models/checkpoints/graphcl/best_model.pt     <- full model (encoder + projector)

Usage:
    python -m src.train.pretrain_graphcl --epochs 30 --save_every 5
    python -m src.train.pretrain_graphcl --epochs 30 --resume   # continua do último checkpoint
"""

import os
import copy
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from src.models.gin_model import GINEEncoder
from src.models.graphcl_model import GraphCLModel, NTXentLoss

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
META_CSV         = "results/embeddings/joint_embeddings_meta.csv"
HIV_GRAPH_DIR    = "data/processed/graphs_hiv"
BRNPDB_GRAPH_DIR = "data/processed/graphs_brnpdb_antiviral"
CKPT_DIR         = "models/checkpoints/graphcl"
CACHE_PATH       = "data/processed/graphcl_joint_cache.pt"
os.makedirs(CKPT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Graph augmentations
# ──────────────────────────────────────────────────────────────

def drop_edges(data: Data, p: float = 0.15) -> Data:
    """Randomly remove edges with probability p. Keeps at least 1 edge."""
    E = data.edge_index.size(1)
    if E == 0:
        return data

    keep_mask = torch.rand(E) > p
    if keep_mask.sum() == 0:
        keep_mask[random.randint(0, E - 1)] = True

    edge_index = data.edge_index[:, keep_mask]
    edge_attr  = data.edge_attr[keep_mask] if data.edge_attr is not None else None

    new_data            = data.clone()
    new_data.edge_index = edge_index
    new_data.edge_attr  = edge_attr
    return new_data


def mask_node_feats(data: Data, p: float = 0.15) -> Data:
    """Randomly zero out atom feature rows with probability p."""
    x        = data.x.clone()
    mask     = torch.rand(x.size(0)) < p
    x[mask]  = 0.0
    new_data = data.clone()
    new_data.x = x
    return new_data


def augment(data: Data) -> Data:
    """Apply one randomly chosen augmentation."""
    choice = random.random()
    if choice < 0.5:
        return drop_edges(data, p=0.15)
    else:
        return mask_node_feats(data, p=0.15)


# ──────────────────────────────────────────────────────────────
# Dataset loader
# ──────────────────────────────────────────────────────────────

def load_all_graphs(meta_csv: str) -> list:
    # Cache: consolidate 41k individual .pt files into one for fast reload
    if os.path.exists(CACHE_PATH):
        print(f"  Loading from cache: {CACHE_PATH} ...")
        graphs = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
        print(f"  {len(graphs)} graphs loaded from cache")
        return graphs

    print(f"  First run: loading individual .pt files (slow, one-time) ...")
    meta = pd.read_csv(meta_csv)

    graphs = []
    missing = 0
    for i, (_, row) in enumerate(meta.iterrows()):
        source    = row["source"]
        jidx      = int(row["joint_idx"])
        brnpdb_id = row.get("brnpdb_id", None)

        if source == "HIV":
            path = os.path.join(HIV_GRAPH_DIR, f"{jidx}.pt")
        else:
            path = os.path.join(BRNPDB_GRAPH_DIR, f"{brnpdb_id}.pt")

        if not os.path.exists(path):
            missing += 1
            continue

        data = torch.load(path, map_location="cpu", weights_only=False)
        graphs.append(data)

        if (i + 1) % 5000 == 0:
            print(f"    {i + 1}/{len(meta)} loaded ...")

    print(f"  Loaded {len(graphs)} graphs  ({missing} missing)")
    print(f"  Saving cache to {CACHE_PATH} ...")
    torch.save(graphs, CACHE_PATH)
    print(f"  Cache saved.")
    return graphs


# ──────────────────────────────────────────────────────────────
# Collate: produce two augmented views per graph in the batch
# ──────────────────────────────────────────────────────────────

from torch_geometric.data import Batch

def contrastive_collate(graph_list):
    """Returns two independently augmented batches from the same graph list."""
    views_i = [augment(g) for g in graph_list]
    views_j = [augment(g) for g in graph_list]
    return Batch.from_data_list(views_i), Batch.from_data_list(views_j)


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch_i, batch_j in loader:
        batch_i = batch_i.to(device)
        batch_j = batch_j.to(device)

        z_i = model(batch_i.x, batch_i.edge_index, batch_i.edge_attr, batch_i.batch)
        z_j = model(batch_j.x, batch_j.edge_index, batch_j.edge_attr, batch_j.batch)

        loss = loss_fn(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load graphs
    print(f"\nLoading joint graph dataset from {META_CSV} ...")
    graphs = load_all_graphs(META_CSV)
    print(f"  Total graphs for contrastive training: {len(graphs)}")

    # 2. DataLoader with contrastive collation
    loader = DataLoader(
        graphs,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=contrastive_collate,
        num_workers=0,
    )

    # 3. Model
    encoder = GINEEncoder(
        in_channels=35,
        edge_dim=9,
        hidden_channels=128,
        num_layers=4,
        dropout=0.2,
    )
    model   = GraphCLModel(encoder, proj_hidden=128, proj_out=64).to(device)
    loss_fn = NTXentLoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Resume state
    start_epoch = 1
    best_loss   = float("inf")

    resume_path = os.path.join(CKPT_DIR, "last_checkpoint.pt")
    if args.resume and os.path.exists(resume_path):
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_loss   = state["best_loss"]
        print(f"  Resumed from epoch {state['epoch']}  (best loss so far: {best_loss:.4f})")
    elif args.init_checkpoint and os.path.exists(args.init_checkpoint):
        encoder.load_state_dict(
            torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        )
        print(f"  Encoder warm-started from {args.init_checkpoint}")

    # Cosine scheduler anchored to total epochs (accounts for resume)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6, last_epoch=start_epoch - 2
    )

    # 4. Training
    end_epoch = start_epoch + args.epochs - 1
    print(f"\nTraining epochs {start_epoch} -> {end_epoch} ...")
    print(f"  batch_size={args.batch_size}  lr={args.lr}  temperature={args.temperature}\n")

    for epoch in range(start_epoch, end_epoch + 1):
        loss = train_one_epoch(model, loader, optimizer, loss_fn, device)
        scheduler.step()

        print(f"Epoch {epoch:>4}/{end_epoch}  loss={loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # Always save last checkpoint (safe resume point)
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "best_loss":  best_loss,
        }, resume_path)

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(),          os.path.join(CKPT_DIR, "best_model.pt"))
            torch.save(model.encoder.state_dict(),  os.path.join(CKPT_DIR, "best_encoder.pt"))
            print(f"         -> best encoder saved (loss={best_loss:.4f})")

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(CKPT_DIR, f"checkpoint_epoch_{epoch}.pt"))

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"Encoder saved to: {os.path.join(CKPT_DIR, 'best_encoder.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",           type=int,   default=30)
    parser.add_argument("--batch_size",       type=int,   default=512)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--temperature",      type=float, default=0.1)
    parser.add_argument("--save_every",       type=int,   default=5)
    parser.add_argument("--resume",           action="store_true",
                        help="Resume from models/checkpoints/graphcl/last_checkpoint.pt")
    parser.add_argument("--init_checkpoint",  type=str,
                        default="models/checkpoints/hiv/best_encoder.pt",
                        help="Warm-start encoder (ignored if --resume is set).")
    args = parser.parse_args()
    main(args)
