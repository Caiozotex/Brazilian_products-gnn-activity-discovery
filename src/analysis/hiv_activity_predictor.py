"""
Multi-Evidence HIV Activity Classifier (MEHAC)

Combines 4 orthogonal signals to predict HIV activity for 147 BrNPDB compounds:

  S1 — mlp_score_norm : GNN direct structural prediction (normalized to [0,1])
  S2 — lp_score_norm  : Label propagation score through k-NN graph (normalized)
  S3 — knn_hiv_frac   : Fraction of immediate graph neighbors that are HIV-active
                        (more local/direct than LP which spreads multi-hop)
  S4 — struct_score   : Functional-group lift score
                        (groups enriched in high-consensus compounds)

Final score = 0.35*S1 + 0.30*S2 + 0.25*S3 + 0.10*S4

Classification threshold = 0.35
Confidence: HIGH>=0.55, MEDIUM 0.40-0.55, LOW 0.30-0.40, VERY_LOW<0.30
"""

import os
import sys
# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import time
import numpy as np
import pandas as pd
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

SCREENING_CSV = "results/tables/brnpdb_consensus_screening.csv"
STRUCT_CSV    = "results/gnnexplainer/brnpdb_structural_features.csv"
GRAPH_GEXF    = "results/similiarity_graph_hiv/hiv_knn_graph.gexf"
OUT_CSV       = "results/tables/brnpdb_hiv_predictions.csv"
OUT_TXT       = "results/tables/brnpdb_hiv_predictions_report.txt"

os.makedirs("results/tables", exist_ok=True)

# Signal weights
W1, W2, W3, W4 = 0.35, 0.30, 0.25, 0.10
THRESHOLD = 0.35


# ── Helpers ───────────────────────────────────────────────────

def minmax_norm(series):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return series * 0.0
    return (series - lo) / (hi - lo)


def compute_knn_signals(G, brnpdb_id_to_node):
    """
    For each BrNPDB compound, compute:
      knn_hiv_frac : fraction of HIV-dataset neighbors that are HIV-active
      n_hiv_nb     : total HIV-dataset neighbors
    """
    result = {}
    for bid, node in brnpdb_id_to_node.items():
        active_count = 0
        hiv_count    = 0
        for nb in G.neighbors(node):
            nb_data = G.nodes[nb]
            dataset = str(nb_data.get("dataset", "HIV")).lower()
            if dataset == "hiv":             # only HIV training nodes contribute
                hiv_count += 1
                # hiv_active is stored as float 0.0/1.0
                val = nb_data.get("hiv_active", 0.0)
                try:
                    if float(val) >= 0.5:
                        active_count += 1
                except (TypeError, ValueError):
                    if str(val).lower() in ("true", "1", "yes", "active"):
                        active_count += 1
        frac = active_count / hiv_count if hiv_count > 0 else 0.0
        result[bid] = (round(frac, 6), hiv_count, active_count)
    return result


def compute_struct_score(screening, struct_df):
    """
    Compute per-compound structural lift score.

    For each functional group g:
      lift(g) = mean_consensus(has g) - mean_consensus(lacks g)

    struct_score = sum of lifts for groups present in compound,
    clipped to [0,1] via min-max normalization.
    """
    if struct_df is None or "functional_groups" not in struct_df.columns:
        return {bid: 0.0 for bid in screening["brnpdb_id"]}

    merged = screening.merge(
        struct_df[["brnpdb_id", "functional_groups"]],
        on="brnpdb_id", how="left"
    )

    # Collect all groups
    all_groups = set()
    for gs in merged["functional_groups"].dropna():
        for g in str(gs).split(";"):
            g = g.strip()
            if g:
                all_groups.add(g)

    # Compute lift per group
    lift = {}
    for g in all_groups:
        has  = merged[merged["functional_groups"].fillna("").str.contains(g, regex=False)]
        lacks = merged[~merged["functional_groups"].fillna("").str.contains(g, regex=False)]
        if len(has) == 0 or len(lacks) == 0:
            lift[g] = 0.0
        else:
            lift[g] = float(has["consensus_score"].mean()) - float(lacks["consensus_score"].mean())

    # Score per compound
    scores = {}
    for _, row in merged.iterrows():
        bid = int(row["brnpdb_id"])
        gs  = str(row.get("functional_groups", ""))
        s   = 0.0
        for g in gs.split(";"):
            g = g.strip()
            if g in lift:
                s += lift[g]
        scores[bid] = s

    # Normalize to [0,1]
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi > lo:
        scores = {k: (v - lo) / (hi - lo) for k, v in scores.items()}
    else:
        scores = {k: 0.0 for k in scores}
    return scores


# ── Main ──────────────────────────────────────────────────────

def main():
    # ── Load screening CSV ────────────────────────────────────
    print("Loading screening data ...")
    screening = pd.read_csv(SCREENING_CSV)
    N = len(screening)
    print(f"  {N} BrNPDB compounds")

    # ── Load structural features ──────────────────────────────
    struct_df = pd.read_csv(STRUCT_CSV) if os.path.exists(STRUCT_CSV) else None

    # ── Normalize mlp + lp to [0,1] ──────────────────────────
    screening["mlp_norm"] = minmax_norm(screening["mlp_score"])
    screening["lp_norm"]  = minmax_norm(screening["lp_score"])

    # ── Load GEXF graph ───────────────────────────────────────
    print("Loading k-NN graph from GEXF (this may take ~30 s) ...")
    t0 = time.time()
    G = nx.read_gexf(GRAPH_GEXF)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges  "
          f"({time.time()-t0:.1f}s)")

    # Map brnpdb_id → graph node id
    # In the GEXF, BrNPDB/antiviral nodes have dataset="Antiviral" and hiv_active=-1.0
    brnpdb_id_to_node = {}
    for node, attrs in G.nodes(data=True):
        dataset = str(attrs.get("dataset", "")).lower()
        if dataset == "antiviral":
            bid = attrs.get("brnpdb_id", None)
            if bid is not None:
                try:
                    brnpdb_id_to_node[int(float(bid))] = node
                except (TypeError, ValueError):
                    pass
    print(f"  {len(brnpdb_id_to_node)} BrNPDB nodes found in graph")

    # ── Compute S3: KNN HIV fraction ──────────────────────────
    print("Computing KNN HIV fractions ...")
    knn_data = compute_knn_signals(G, brnpdb_id_to_node)

    # ── Compute S4: structural lift score ─────────────────────
    print("Computing structural lift scores ...")
    struct_scores = compute_struct_score(screening, struct_df)

    # ── Build final predictions ───────────────────────────────
    print("Building predictions ...")
    records = []
    for _, row in screening.iterrows():
        bid  = int(row["brnpdb_id"])
        s1   = float(row["mlp_norm"])
        s2   = float(row["lp_norm"])
        knn  = knn_data.get(bid, (0.0, 0, 0))
        s3   = knn[0]
        s4   = struct_scores.get(bid, 0.0)

        final = W1*s1 + W2*s2 + W3*s3 + W4*s4

        # Classification
        predicted = "ACTIVE" if final >= THRESHOLD else "INACTIVE"
        if final >= 0.55:
            confidence = "HIGH"
        elif final >= 0.40:
            confidence = "MEDIUM"
        elif final >= 0.30:
            confidence = "LOW"
        else:
            confidence = "VERY_LOW"

        # Signal agreement (how many of the 3 main signals support activity)
        # S1: mlp_norm >= 0.5; S2: lp_norm >= 0.5; S3: knn_hiv_frac >= 0.05
        agree = (
            int(s1 >= 0.50) +
            int(s2 >= 0.50) +
            int(s3 >= 0.05) +
            int(s4 >= 0.50)
        )

        records.append({
            "brnpdb_id"         : bid,
            "common_name"       : row["common_name"],
            "community"         : row.get("community", ""),
            "mlp_score"         : round(float(row["mlp_score"]), 6),
            "lp_score"          : round(float(row["lp_score"]),  6),
            "consensus_score"   : round(float(row["consensus_score"]), 6),
            "S1_mlp_norm"       : round(s1, 4),
            "S2_lp_norm"        : round(s2, 4),
            "S3_knn_hiv_frac"   : round(s3, 4),
            "S4_struct_score"   : round(s4, 4),
            "n_hiv_neighbors"   : knn[1],
            "n_active_neighbors": knn[2],
            "final_score"       : round(final, 4),
            "predicted_activity": predicted,
            "confidence"        : confidence,
            "n_signals_active"  : agree,
        })

    df = pd.DataFrame(records).sort_values("final_score", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    # ── Summary stats ─────────────────────────────────────────
    active   = df[df["predicted_activity"] == "ACTIVE"]
    inactive = df[df["predicted_activity"] == "INACTIVE"]

    print("\n" + "=" * 70)
    print("MEHAC PREDICTIONS — BrNPDB HIV Activity")
    print("=" * 70)
    print(f"Total compounds : {len(df)}")
    print(f"Predicted ACTIVE: {len(active)}  ({len(active)/len(df)*100:.1f}%)")
    print(f"  HIGH confidence   : {(active['confidence']=='HIGH').sum()}")
    print(f"  MEDIUM confidence : {(active['confidence']=='MEDIUM').sum()}")
    print(f"  LOW confidence    : {(active['confidence']=='LOW').sum()}")
    print(f"Predicted INACTIVE: {len(inactive)}")

    print("\n── Top 20 (predicted ACTIVE + borderline) ──")
    print(f"{'Rank':>4}  {'BrNPDB':>7}  {'Name':<42}  {'final':>6}  {'conf':<9}  S1   S2   S3   S4")
    print("-" * 100)
    for i, row in df.head(30).iterrows():
        mark = "✓" if row["predicted_activity"] == "ACTIVE" else "·"
        name = str(row["common_name"])[:40]
        print(f"{i+1:>4}  {row['brnpdb_id']:>7}  {name:<42}  "
              f"{row['final_score']:>6.4f}  {row['confidence']:<9}  "
              f"{row['S1_mlp_norm']:.2f} {row['S2_lp_norm']:.2f} "
              f"{row['S3_knn_hiv_frac']:.2f} {row['S4_struct_score']:.2f}  {mark}")

    print("\n── Summary by signal agreement ──")
    for n_sig in [4, 3, 2, 1, 0]:
        sub = df[df["n_signals_active"] == n_sig]
        if len(sub) == 0:
            continue
        n_active = (sub["predicted_activity"] == "ACTIVE").sum()
        print(f"  {n_sig} signals agree → {len(sub):>3} compounds  ({n_active} predicted active)")

    # ── Readable report ───────────────────────────────────────
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("MEHAC — Multi-Evidence HIV Activity Classifier\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Algorithm: final_score = {W1}*S1 + {W2}*S2 + {W3}*S3 + {W4}*S4\n")
        f.write("  S1 = mlp_score normalized  (GNN direct structural prediction)\n")
        f.write("  S2 = lp_score normalized   (label propagation through k-NN graph)\n")
        f.write("  S3 = knn_hiv_frac          (fraction HIV-active immediate neighbors)\n")
        f.write("  S4 = struct_score          (functional-group lift)\n")
        f.write(f"Threshold: {THRESHOLD}  (>= → ACTIVE)\n\n")
        f.write(f"Total compounds : {len(df)}\n")
        f.write(f"Predicted ACTIVE: {len(active)}\n")
        f.write(f"Predicted INACTIVE: {len(inactive)}\n\n")

        f.write("─── PREDICTED ACTIVE ──────────────────────────────────────────────────\n\n")
        for _, row in active.sort_values("final_score", ascending=False).iterrows():
            name = str(row["common_name"]).encode("ascii", "replace").decode("ascii")
            f.write(f"BrNPDB {row['brnpdb_id']:>6}  [{row['confidence']:<9}]  "
                    f"score={row['final_score']:.4f}  "
                    f"S1={row['S1_mlp_norm']:.3f}  S2={row['S2_lp_norm']:.3f}  "
                    f"S3={row['S3_knn_hiv_frac']:.3f}  S4={row['S4_struct_score']:.3f}\n")
            f.write(f"  Name     : {name}\n")
            f.write(f"  Community: {row['community']}\n")
            f.write(f"  HIV neighbors: {row['n_active_neighbors']}/{row['n_hiv_neighbors']} active\n")
            f.write(f"  Signals agree: {row['n_signals_active']}/4\n\n")

        f.write("\n─── PREDICTED INACTIVE (borderline, score 0.25–0.35) ──────────────────\n\n")
        border = inactive[inactive["final_score"] >= 0.25].sort_values("final_score", ascending=False)
        for _, row in border.iterrows():
            name = str(row["common_name"]).encode("ascii", "replace").decode("ascii")
            f.write(f"BrNPDB {row['brnpdb_id']:>6}  [BORDERLINE]  "
                    f"score={row['final_score']:.4f}\n")
            f.write(f"  Name: {name}\n\n")

    print(f"\nSaved report: {OUT_TXT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
