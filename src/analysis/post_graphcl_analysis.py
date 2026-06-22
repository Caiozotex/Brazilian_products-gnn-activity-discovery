"""
Post-GraphCL analysis pipeline. Runs automatically after pretrain_graphcl.py.

Steps:
  1. Generate GraphCL graph-level embeddings
  2. Build hybrid similarity graph (alpha=0.2, threshold=0.84)
  3. Check graph connectivity
  4. Louvain community detection (weighted by similarity)
  5. Per-community report: brnpdb nodes, HIV-active, HIV-inactive, unknown
"""

import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from torch_geometric.loader import DataLoader as PygLoader
import warnings
warnings.filterwarnings("ignore")

from src.models.gin_model import GINEEncoder, GraphPooling
from src.analysis.louvain import louvain as louvain_from_scratch, modularity

# ── Config ────────────────────────────────────────────────────
ALPHA          = 0.2
THRESHOLD      = 0.84
CHUNK          = 256

ENCODER_CKPT   = "models/checkpoints/graphcl/best_encoder.pt"
META_PATH      = "results/embeddings/joint_embeddings_meta.csv"
JOINT_CSV      = "data/processed/joint_data.csv"
HIV_GRAPH_DIR  = "data/processed/graphs_hiv"
BRNPDB_DIR     = "data/processed/graphs_brnpdb_antiviral"

OUT_DIR        = "results/graphcl"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Step 1: Generate GraphCL embeddings ───────────────────────

def generate_embeddings():
    print("\n[1/5] Generating GraphCL embeddings ...")
    encoder = GINEEncoder(in_channels=35, edge_dim=9, hidden_channels=128,
                          num_layers=4, dropout=0.2)
    encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=DEVICE, weights_only=True))
    encoder.to(DEVICE).eval()
    pool = GraphPooling()

    meta = pd.read_csv(META_PATH)
    graphs, kept_idx = [], []

    for _, row in meta.iterrows():
        jidx = int(row["joint_idx"])
        path = (os.path.join(HIV_GRAPH_DIR, f"{jidx}.pt") if row["source"] == "HIV"
                else os.path.join(BRNPDB_DIR, f"{int(float(row['brnpdb_id']))}.pt"))
        if not os.path.exists(path):
            continue
        g = torch.load(path, map_location="cpu", weights_only=False)
        # Keep only the attributes needed for inference so HIV and BrNPDB graphs
        # can be batched together (HIV graphs carry extra fields like 'activity').
        from torch_geometric.data import Data
        g = Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr)
        graphs.append(g)
        kept_idx.append(jidx)

    loader = PygLoader(graphs, batch_size=512, shuffle=False)

    @torch.no_grad()
    def embed():
        all_emb = []
        for batch in loader:
            batch = batch.to(DEVICE)
            node_emb  = encoder(batch.x, batch.edge_index, batch.edge_attr)
            graph_emb = pool(node_emb, batch.batch)
            all_emb.append(graph_emb.cpu())
        return torch.cat(all_emb, dim=0).numpy()

    emb = embed()
    print(f"  {emb.shape[0]} embeddings of dim {emb.shape[1]}")

    # Save
    np.save(os.path.join(OUT_DIR, "graphcl_embeddings.npy"), emb)
    meta_out = meta.set_index("joint_idx").loc[kept_idx].copy().reset_index()
    meta_out["emb_row"] = range(len(meta_out))
    meta_out.to_csv(os.path.join(OUT_DIR, "graphcl_embeddings_meta.csv"), index=False)

    return emb, meta_out


# ── Step 2: Load Morgan fingerprints ──────────────────────────

def load_fingerprints(meta_out):
    print("\n[2/5] Computing Morgan fingerprints ...")
    joint = pd.read_csv(JOINT_CSV)
    idx_to_smiles = joint.set_index("joint_idx")["smiles"].to_dict()

    fps = []
    for jidx in meta_out["joint_idx"]:
        smi = idx_to_smiles.get(jidx, None)
        mol = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) else None
        if mol is None:
            fps.append(np.zeros(2048, dtype=np.float32))
            continue
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros(2048, dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        fps.append(arr)

    print(f"  {len(fps)} fingerprints ready")
    return np.stack(fps)


# ── Step 3: Build graph ────────────────────────────────────────

def build_graph(emb, fps_mat):
    print(f"\n[3/5] Building graph (alpha={ALPHA}, threshold={THRESHOLD}) ...")
    N = emb.shape[0]

    # Centered cosine
    cc = emb - emb.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(cc, axis=1, keepdims=True)
    cc = cc / np.where(norms == 0, 1.0, norms)

    rows_i, rows_j, weights = [], [], []
    t0 = time.time()

    for chunk_start in range(0, N, CHUNK):
        chunk_end  = min(chunk_start + CHUNK, N)
        cc_chunk   = cc[chunk_start:chunk_end]
        tan_chunk  = fps_mat[chunk_start:chunk_end]

        cc_block   = cc_chunk @ cc.T

        inter      = tan_chunk @ fps_mat.T
        cnt_c      = tan_chunk.sum(axis=1, keepdims=True)
        cnt_m      = fps_mat.sum(axis=1, keepdims=True).T
        union      = cnt_c + cnt_m - inter
        tan_block  = np.where(union > 0, inter / union, 0.0)

        sim_block  = ALPHA * tan_block + (1 - ALPHA) * cc_block

        for local_i, global_i in enumerate(range(chunk_start, chunk_end)):
            sim_row = sim_block[local_i, global_i + 1:]
            js      = np.where(sim_row >= THRESHOLD)[0] + global_i + 1
            if len(js) == 0:
                continue
            rows_i.extend([global_i] * len(js))
            rows_j.extend(js.tolist())
            weights.extend(sim_row[js - global_i - 1].tolist())

        if (chunk_start // CHUNK) % 20 == 0:
            elapsed = time.time() - t0
            pct = chunk_end / N * 100
            print(f"  {chunk_end}/{N} ({pct:.1f}%)  edges: {len(rows_i):,}  "
                  f"elapsed: {elapsed/60:.1f}min")

    print(f"  Total edges (undirected): {len(rows_i):,}")
    return np.array(rows_i), np.array(rows_j), np.array(weights), N


# ── Step 4: Connectivity check ─────────────────────────────────

def check_connectivity(rows_i, rows_j, weights, N):
    print("\n[4/5] Checking connectivity ...")
    data = np.concatenate([weights, weights])
    row  = np.concatenate([rows_i, rows_j]).astype(np.int32)
    col  = np.concatenate([rows_j, rows_i]).astype(np.int32)
    adj  = sp.csr_matrix((data, (row, col)), shape=(N, N))

    n_comp, labels = csgraph.connected_components(adj, directed=False)
    comp_sizes     = np.bincount(labels)
    largest        = comp_sizes.max()
    isolated       = (comp_sizes == 1).sum()

    print(f"\n  {'='*45}")
    print(f"  Nodes              : {N:,}")
    print(f"  Edges              : {len(rows_i):,}")
    print(f"  Connected          : {'YES' if n_comp == 1 else 'NO'}")
    print(f"  Components         : {n_comp:,}")
    print(f"  Largest component  : {largest:,} ({largest/N*100:.1f}%)")
    print(f"  Isolated nodes     : {isolated:,} ({isolated/N*100:.2f}%)")
    print(f"  {'='*45}")

    return adj, labels, n_comp


# ── Step 5: Louvain + report ────────────────────────────────────

def louvain_and_report(rows_i, rows_j, weights, N, labels_cc, meta_out):
    print("\n[5/5] Louvain community detection (custom implementation) ...")
    t_louv = time.time()

    comm_labels = louvain_from_scratch(N, rows_i, rows_j, weights,
                                       max_phases=20, max_sweeps=100,
                                       seed=42, verbose=True)

    Q = modularity(N, rows_i, rows_j, weights, comm_labels)
    elapsed_louv = (time.time() - t_louv) / 60
    n_comms = len(set(comm_labels.tolist()))
    print(f"  Louvain finished in {elapsed_louv:.1f} min")
    print(f"  Communities found: {n_comms:,}")
    print(f"  Modularity Q = {Q:.4f}")

    # meta_out already contains HIV_active, source, brnpdb_id, common_name
    meta_full = meta_out.copy().reset_index(drop=True)
    meta_full["community"] = comm_labels

    # Per-community report (group by community label)
    rows = []
    for cid, sub in meta_full.groupby("community"):
        brnpdb  = (sub["source"] == "brnpdb").sum()
        hiv_act = (sub["HIV_active"] == 1.0).sum()
        hiv_ina = (sub["HIV_active"] == 0.0).sum()
        hiv_unk = sub["HIV_active"].isna().sum()
        rows.append({
            "community_id":    int(cid),
            "size":            len(sub),
            "brnpdb_nodes":    int(brnpdb),
            "hiv_active":      int(hiv_act),
            "hiv_inactive":    int(hiv_ina),
            "hiv_unknown":     int(hiv_unk),
            "brnpdb_pct":      round(brnpdb / len(sub) * 100, 2),
            "hiv_active_rate": round(hiv_act / (hiv_act + hiv_ina) * 100, 2)
                               if (hiv_act + hiv_ina) > 0 else None,
        })

    report = pd.DataFrame(rows).sort_values("size", ascending=False)
    report_path = os.path.join(OUT_DIR, "community_report.csv")
    report.to_csv(report_path, index=False)

    # Node-level assignment
    node_path = os.path.join(OUT_DIR, "node_communities.csv")
    meta_full[["joint_idx", "source", "brnpdb_id", "common_name",
               "HIV_active", "community"]].to_csv(node_path, index=False)

    # Console summary
    print(f"\n  Top 15 communities by size:")
    print(f"  {'ID':>5}  {'Size':>7}  {'BrNPDB':>7}  {'HIV+':>6}  {'HIV-':>7}  {'Unknown':>8}  {'HIV+%':>7}")
    print(f"  {'-'*58}")
    for _, r in report.head(15).iterrows():
        rate = f"{r['hiv_active_rate']:.1f}%" if r['hiv_active_rate'] is not None else "  N/A"
        print(f"  {int(r.community_id):>5}  {int(r['size']):>7,}  {int(r.brnpdb_nodes):>7}  "
              f"{int(r.hiv_active):>6}  {int(r.hiv_inactive):>7}  {int(r.hiv_unknown):>8}  {rate:>7}")

    print(f"\n  Full report -> {report_path}")
    print(f"  Node assignments -> {node_path}")

    # BrNPDB nodes summary
    brnpdb_nodes = meta_full[meta_full["source"] == "brnpdb"][
        ["joint_idx", "brnpdb_id", "common_name", "community"]
    ].copy()
    brnpdb_nodes = brnpdb_nodes.merge(
        report[["community_id", "size", "hiv_active", "hiv_inactive", "hiv_active_rate"]],
        left_on="community", right_on="community_id", how="left"
    )
    brnpdb_path = os.path.join(OUT_DIR, "brnpdb_community_placement.csv")
    brnpdb_nodes.to_csv(brnpdb_path, index=False)
    print(f"  BrNPDB placement -> {brnpdb_path}")


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.time()
    print("=" * 50)
    print("Post-GraphCL Analysis Pipeline")
    print("=" * 50)

    emb, meta_out    = generate_embeddings()
    fps_mat          = load_fingerprints(meta_out)
    rows_i, rows_j, weights, N = build_graph(emb, fps_mat)
    adj, cc_labels, n_comp     = check_connectivity(rows_i, rows_j, weights, N)
    louvain_and_report(rows_i, rows_j, weights, N, cc_labels, meta_out)

    elapsed = (time.time() - t_start) / 60
    print(f"\nTotal analysis time: {elapsed:.1f} min")
    print("Done.")
