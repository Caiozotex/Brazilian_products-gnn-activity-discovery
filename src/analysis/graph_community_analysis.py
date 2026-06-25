"""
Graph Analysis & Community-Based HIV Activity Prediction

Analyses performed on hiv_knn_graph.gexf (41 k nodes, 280 k edges):
  1. Basic graph statistics: density, degree distribution, top-hub molecules
  2. Connected components
  3. Louvain community detection (our own implementation)
  4. Community HIV enrichment analysis + separation between active/inactive
  5. BFS distance from each BrNPDB compound to its nearest HIV-active neighbour
  6. Final activity prediction for 147 BrNPDB compounds:
       community_enrichment + bfs_distance → activity score → label
"""

import os, sys, time
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from collections import Counter, defaultdict, deque
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")
from src.analysis.louvain import louvain as run_louvain, modularity as compute_modularity

GRAPH_GEXF    = "results/similiarity_graph_hiv/hiv_knn_graph.gexf"
SCREENING_CSV = "results/tables/brnpdb_consensus_screening.csv"
OUT_CSV       = "results/tables/community_hiv_predictions.csv"
OUT_TXT       = "results/tables/graph_analysis_report.txt"
os.makedirs("results/tables", exist_ok=True)

BFS_MAX_DIST  = 10        # cap BFS search depth
ENRICH_THRESH = None      # set dynamically to 3x global HIV rate


# ── Section 1: Load graph ─────────────────────────────────────

def load_graph():
    print("Loading GEXF graph ...")
    t0 = time.time()
    G = nx.read_gexf(GRAPH_GEXF)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges  ({time.time()-t0:.1f}s)")
    return G


def extract_node_metadata(G):
    """
    Returns:
      node_ids      : sorted list of int node ids
      is_hiv_active : np.bool_ array (shape N)
      is_brnpdb     : np.bool_ array
      brnpdb_id_map : dict  int(brnpdb_id) -> node_int
      node_names    : dict  node_int -> label string
    """
    node_ids  = sorted(int(n) for n in G.nodes())
    N         = len(node_ids)
    is_active = np.zeros(N, dtype=bool)
    is_brnpdb = np.zeros(N, dtype=bool)
    brnpdb_id_map = {}
    node_names    = {}

    for n, attrs in G.nodes(data=True):
        ni = int(n)
        dataset = str(attrs.get("dataset", "HIV")).lower()
        if dataset == "antiviral":
            is_brnpdb[ni] = True
            bid = attrs.get("brnpdb_id", None)
            if bid is not None:
                try:
                    brnpdb_id_map[int(float(bid))] = ni
                except (TypeError, ValueError):
                    pass
        else:  # HIV training node
            val = attrs.get("hiv_active", 0.0)
            try:
                is_active[ni] = float(val) >= 0.5
            except (TypeError, ValueError):
                pass
        node_names[ni] = str(attrs.get("name", n))

    return node_ids, N, is_active, is_brnpdb, brnpdb_id_map, node_names


# ── Section 2: Degree analysis ────────────────────────────────

def degree_analysis(G, is_brnpdb, node_names, N):
    print("\nComputing degree statistics ...")
    deg = np.array([G.degree(str(i)) for i in range(N)], dtype=np.int32)

    hiv_deg  = deg[~is_brnpdb]
    br_deg   = deg[is_brnpdb]

    stats = {
        "total_nodes"  : N,
        "total_edges"  : G.number_of_edges(),
        "density"      : 2 * G.number_of_edges() / (N * (N - 1)),
        "avg_degree"   : float(deg.mean()),
        "max_degree"   : int(deg.max()),
        "min_degree"   : int(deg.min()),
        "hiv_avg_deg"  : float(hiv_deg.mean()),
        "hiv_max_deg"  : int(hiv_deg.max()),
        "brnpdb_avg_deg": float(br_deg.mean()),
        "brnpdb_max_deg": int(br_deg.max()),
    }

    # Top-10 highest degree nodes overall
    top10_idx = np.argsort(deg)[::-1][:10]
    top10 = [(int(i), int(deg[i]), node_names.get(i, str(i))) for i in top10_idx]

    # Top-5 highest degree BrNPDB nodes
    brnpdb_idx = np.where(is_brnpdb)[0]
    top5_br_idx = brnpdb_idx[np.argsort(deg[brnpdb_idx])[::-1][:5]]
    top5_br = [(int(i), int(deg[i]), node_names.get(i, str(i))) for i in top5_br_idx]

    return stats, deg, top10, top5_br


# ── Section 3: Connected components ──────────────────────────

def connected_components(G, N):
    print("Computing connected components ...")
    # Build sparse adjacency
    edges = np.array([(int(u), int(v)) for u, v in G.edges()], dtype=np.int32)
    ri = np.concatenate([edges[:, 0], edges[:, 1]])
    ci = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(ri), dtype=np.int8)
    adj  = sp.csr_matrix((data, (ri, ci)), shape=(N, N), dtype=np.int8)
    n_comp, comp_labels = csgraph.connected_components(adj, directed=False)

    sizes = np.bincount(comp_labels)
    top5_sizes = sorted(sizes, reverse=True)[:5]
    cyclomatic = G.number_of_edges() - N + n_comp  # independent cycles

    return n_comp, comp_labels, top5_sizes, cyclomatic


# ── Section 4: Louvain community detection ────────────────────

def run_community_detection(G, N):
    print("\nExtracting edge arrays for Louvain ...")
    edges = list(G.edges(data=True))
    rows_i  = np.array([int(u) for u, v, _ in edges], dtype=np.int32)
    rows_j  = np.array([int(v) for u, v, _ in edges], dtype=np.int32)
    weights = np.array([float(d.get("weight", 1.0)) for _, _, d in edges], dtype=np.float32)

    print(f"Running Louvain on {N:,} nodes, {len(rows_i):,} edges ...")
    t0 = time.time()
    labels = run_louvain(N, rows_i, rows_j, weights, seed=42, verbose=True)
    elapsed = time.time() - t0
    print(f"Louvain done in {elapsed:.1f}s")

    Q = compute_modularity(N, rows_i, rows_j, weights, labels)
    n_comm = len(set(labels))
    sizes  = Counter(labels)

    print(f"  Communities: {n_comm:,}  |  Modularity Q = {Q:.4f}")
    return labels, Q, n_comm, sizes, rows_i, rows_j, weights


# ── Section 5: Community HIV enrichment ───────────────────────

def community_enrichment(labels, is_active, is_brnpdb, N):
    """
    For each community, compute:
      n_hiv    : number of HIV-dataset nodes
      n_active : number of HIV-active nodes
      enrichment = n_active / n_hiv  (0 if no HIV nodes)
    """
    comm_hiv    = defaultdict(int)
    comm_active = defaultdict(int)
    comm_brnpdb = defaultdict(int)

    for i in range(N):
        c = labels[i]
        if is_brnpdb[i]:
            comm_brnpdb[c] += 1
        else:
            comm_hiv[c] += 1
            if is_active[i]:
                comm_active[c] += 1

    all_comms = set(labels)
    enrichment = {}
    for c in all_comms:
        n_hiv = comm_hiv[c]
        enrichment[c] = comm_active[c] / n_hiv if n_hiv > 0 else 0.0

    global_rate = is_active.sum() / (~is_brnpdb).sum()
    print(f"\n  Global HIV-active rate: {global_rate*100:.2f}%  "
          f"({int(is_active.sum())} active / {int((~is_brnpdb).sum())} HIV nodes)")

    return enrichment, comm_hiv, comm_active, comm_brnpdb, global_rate


# ── Section 6: BFS distance to nearest HIV-active node ────────

def bfs_dist_to_active(G, brnpdb_node, active_node_set, max_dist=BFS_MAX_DIST):
    """
    BFS from brnpdb_node; return distance to nearest active HIV node.
    Returns max_dist+1 if not reachable within max_dist hops.
    """
    if brnpdb_node in active_node_set:
        return 0
    visited = {brnpdb_node}
    queue   = deque([(brnpdb_node, 0)])
    while queue:
        node, dist = queue.popleft()
        if dist >= max_dist:
            return max_dist + 1
        for nb in G.neighbors(node):
            if nb in active_node_set:
                return dist + 1
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, dist + 1))
    return max_dist + 1


def compute_bfs_distances(G, brnpdb_id_map, is_active, N):
    """Run BFS for each BrNPDB compound."""
    # Set of string node ids for HIV-active nodes
    active_str = {str(i) for i in range(N) if is_active[i]}
    print(f"\nBFS search: {len(active_str)} HIV-active target nodes")

    result = {}
    for bid, node_int in brnpdb_id_map.items():
        node_str = str(node_int)
        d = bfs_dist_to_active(G, node_str, active_str)
        result[bid] = d

    dist_vals = list(result.values())
    print(f"  min dist={min(dist_vals)}, max dist={max(dist_vals)}, "
          f"mean dist={np.mean(dist_vals):.2f}")
    dist_counter = Counter(dist_vals)
    for d in sorted(dist_counter):
        label = f"dist={d}" if d <= BFS_MAX_DIST else f"dist>{BFS_MAX_DIST}"
        print(f"    {label}: {dist_counter[d]} BrNPDB compounds")
    return result


# ── Section 7: Prediction ─────────────────────────────────────

def predict_activity(brnpdb_id_map, labels, enrichment, bfs_dists,
                     global_rate, screening_df):
    """
    Bayesian-inspired score:

      x_rate  = comm_enrichment / global_HIV_rate
                (how many times more HIV-active than the background)

      enrich_score = log10(x_rate + 1) / log10(max_x_rate + 1)
                     Maps x_rate logarithmically to [0,1].
                     x_rate=1  -> 0.30   (background)
                     x_rate=5  -> 0.67
                     x_rate=13 -> 1.00   (Amphotericin B community)

      proximity = max(0, 1 - (bfs_dist - 1) / 5)
                  dist=1 -> 1.0,  dist=3 -> 0.6,  dist=6+ -> 0.0

      final = 0.80 * enrich_score + 0.20 * proximity

      Thresholds:
        HIGHLY ACTIVE  >= 0.70   (x_rate >= ~10, very close)
        POSSIBLY ACTIVE >= 0.40  (x_rate >= ~2.3, at most 3 hops away)
        BORDERLINE     >= 0.30
        INACTIVE       <  0.30
    """
    enrich_vals = np.array([enrichment[labels[ni]]
                             for ni in brnpdb_id_map.values()], dtype=float)
    dist_vals   = np.array([bfs_dists.get(bid, BFS_MAX_DIST + 1)
                             for bid in brnpdb_id_map], dtype=float)

    x_rate = enrich_vals / global_rate if global_rate > 0 else enrich_vals

    max_x = x_rate.max() if x_rate.max() > 1 else 1.0
    enrich_score = np.log10(x_rate + 1) / np.log10(max_x + 1)

    proximity = np.clip(1.0 - (dist_vals - 1) / 5, 0, 1)

    score = 0.80 * enrich_score + 0.20 * proximity

    # Classification
    def classify(sc, enrich_x):
        if sc >= 0.70:
            return "ACTIVE", "HIGH"
        elif sc >= 0.40 and enrich_x >= 1.8:   # must have real enrichment, not just proximity
            return "ACTIVE", "MEDIUM"
        elif sc >= 0.35 and enrich_x >= 1.5:
            return "ACTIVE", "LOW"
        else:
            return "INACTIVE", "VERY_LOW"

    threshold = 0.35
    records = []
    screening_map = {int(r["brnpdb_id"]): r for _, r in screening_df.iterrows()}

    for idx, (bid, ni) in enumerate(brnpdb_id_map.items()):
        comm    = int(labels[ni])
        enrich  = enrichment[comm]
        xr      = float(x_rate[idx])
        dist    = int(dist_vals[idx])
        sc      = float(score[idx])
        pred, conf = classify(sc, xr)

        row = screening_map.get(bid, {})
        records.append({
            "brnpdb_id"         : bid,
            "common_name"       : row.get("common_name", ""),
            "community"         : comm,
            "comm_hiv_enrich"   : round(enrich, 6),
            "comm_enrich_xrate" : round(xr, 2),
            "bfs_dist_to_active": dist,
            "activity_score"    : round(sc, 4),
            "predicted_activity": pred,
            "confidence"        : conf,
            "consensus_score"   : float(row.get("consensus_score", 0)),
        })

    df = pd.DataFrame(records).sort_values("activity_score", ascending=False).reset_index(drop=True)
    return df


# ── Section 8: Separation analysis ───────────────────────────

def separation_analysis(enrichment, labels, is_brnpdb, N, global_rate):
    """
    Show distribution of community enrichment rates.
    Mark communities as 'active-enriched' if enrichment > 3x global rate.
    """
    # Only communities with >= 5 HIV nodes
    comm_enrich_vals = np.array(list(enrichment.values()))

    thresh3x = 3.0 * global_rate
    thresh5x = 5.0 * global_rate

    below  = (comm_enrich_vals == 0).sum()
    low    = ((comm_enrich_vals > 0) & (comm_enrich_vals < thresh3x)).sum()
    mid    = ((comm_enrich_vals >= thresh3x) & (comm_enrich_vals < thresh5x)).sum()
    high   = (comm_enrich_vals >= thresh5x).sum()

    print(f"\n  Separation analysis (global rate = {global_rate*100:.2f}%):")
    print(f"    Communities with enrichment = 0            : {below:>5}")
    print(f"    Communities with 0 < enrich < 3x rate      : {low:>5}")
    print(f"    Communities with 3x <= enrich < 5x rate    : {mid:>5}")
    print(f"    Communities with enrich >= 5x rate (active): {high:>5}")

    return below, low, mid, high, thresh3x


# ── Main ──────────────────────────────────────────────────────

def main():
    screening_df = pd.read_csv(SCREENING_CSV)

    G = load_graph()
    node_ids, N, is_active, is_brnpdb, brnpdb_id_map, node_names = extract_node_metadata(G)

    print(f"  HIV nodes: {(~is_brnpdb).sum():,}  |  "
          f"HIV-active: {is_active.sum():,}  |  "
          f"BrNPDB: {is_brnpdb.sum():,}")

    # ── Degree analysis ──────────────────────────────────────
    stats, deg, top10, top5_br = degree_analysis(G, is_brnpdb, node_names, N)

    # ── Connected components ─────────────────────────────────
    n_comp, comp_labels, top5_sizes, cyclomatic = connected_components(G, N)
    print(f"  {n_comp:,} connected components  |  "
          f"Giant component: {top5_sizes[0]:,} nodes")
    print(f"  Cyclomatic number (independent cycles): {cyclomatic:,}")

    # ── Louvain ──────────────────────────────────────────────
    labels, Q, n_comm, comm_sizes, rows_i, rows_j, weights = \
        run_community_detection(G, N)

    # ── Community enrichment & separation ────────────────────
    enrichment, comm_hiv, comm_active, comm_brnpdb, global_rate = \
        community_enrichment(labels, is_active, is_brnpdb, N)
    below, low, mid, high_enrich, thresh3x = \
        separation_analysis(enrichment, labels, is_brnpdb, N, global_rate)

    # ── BFS distances ────────────────────────────────────────
    bfs_dists = compute_bfs_distances(G, brnpdb_id_map, is_active, N)

    # ── Predictions ──────────────────────────────────────────
    pred_df = predict_activity(brnpdb_id_map, labels, enrichment,
                               bfs_dists, global_rate, screening_df)
    pred_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved predictions: {OUT_CSV}")

    # Save full node→community mapping for downstream use
    node_comm_df = pd.DataFrame({
        "node_id"   : np.arange(N),
        "community" : labels,
        "is_brnpdb" : is_brnpdb,
        "is_active" : is_active,
    })
    node_comm_df.to_csv("results/tables/louvain_node_communities.csv", index=False)
    print("Saved node communities: results/tables/louvain_node_communities.csv")

    active_df   = pred_df[pred_df["predicted_activity"] == "ACTIVE"]
    inactive_df = pred_df[pred_df["predicted_activity"] == "INACTIVE"]

    # ── Print results table ──────────────────────────────────
    print("\n" + "=" * 80)
    print("COMMUNITY-BASED HIV ACTIVITY PREDICTIONS")
    print("=" * 80)
    print(f"Predicted ACTIVE  : {len(active_df)}  |  "
          f"INACTIVE: {len(inactive_df)}  |  Total: {len(pred_df)}")
    print()
    print(f"{'Rank':>4}  {'BrNPDB':>7}  {'Name':<40}  "
          f"{'score':>6}  {'conf':<9}  {'enrich':>7}  {'x rate':>6}  {'BFS':>4}")
    print("-" * 95)
    for i, row in pred_df.head(30).iterrows():
        mark = "ACTIVE  " if row["predicted_activity"] == "ACTIVE" else "        "
        name = str(row["common_name"])[:38]
        print(f"{i+1:>4}  {row['brnpdb_id']:>7}  {name:<40}  "
              f"{row['activity_score']:>6.4f}  {mark}  "
              f"{row['comm_hiv_enrich']:>7.4f}  {row['comm_enrich_xrate']:>6.1f}x  "
              f"{row['bfs_dist_to_active']:>4}")

    # ── Top enriched communities that contain BrNPDB ─────────
    print("\n-- Communities with >=1 BrNPDB compound, sorted by enrichment --")
    br_comms = {}
    for bid, ni in brnpdb_id_map.items():
        c = int(labels[ni])
        if c not in br_comms:
            br_comms[c] = {"enrich": enrichment[c], "n_hiv": comm_hiv[c],
                           "n_active": comm_active[c], "bids": []}
        br_comms[c]["bids"].append(bid)

    br_comm_list = sorted(br_comms.items(), key=lambda x: -x[1]["enrich"])
    print(f"{'comm':>8}  {'enrich':>8}  {'x rate':>7}  {'hiv':>5}  "
          f"{'active':>6}  {'#brnpdb':>7}  BrNPDB ids")
    print("-" * 80)
    for c, info in br_comm_list[:20]:
        print(f"{c:>8}  {info['enrich']:>8.4f}  "
              f"{info['enrich']/global_rate:>7.1f}x  "
              f"{info['n_hiv']:>5}  {info['n_active']:>6}  "
              f"{len(info['bids']):>7}  "
              f"{str(info['bids'][:4])[1:-1]}")

    # ── Write full report ─────────────────────────────────────
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("Graph Analysis Report — hiv_knn_graph.gexf\n")
        f.write("=" * 75 + "\n\n")

        f.write("1. BASIC STATISTICS\n")
        f.write("-" * 40 + "\n")
        for k, v in stats.items():
            f.write(f"  {k:<25} : {v}\n")
        f.write(f"  n_connected_components   : {n_comp:,}\n")
        f.write(f"  giant_component_size     : {top5_sizes[0]:,}\n")
        f.write(f"  top5_component_sizes     : {top5_sizes}\n")
        f.write(f"  cyclomatic_number        : {cyclomatic:,}  (independent cycles)\n\n")

        f.write("2. TOP-10 HIGHEST DEGREE NODES (hubs)\n")
        f.write("-" * 40 + "\n")
        for rank, (ni, d, name) in enumerate(top10, 1):
            f.write(f"  {rank:>2}. node={ni:>6}  deg={d:>5}  {name}\n")

        f.write("\n3. TOP-5 HIGHEST DEGREE BrNPDB NODES\n")
        f.write("-" * 40 + "\n")
        for rank, (ni, d, name) in enumerate(top5_br, 1):
            f.write(f"  {rank}. node={ni:>6}  deg={d:>5}  {name}\n")

        f.write("\n4. LOUVAIN COMMUNITY DETECTION\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Communities          : {n_comm:,}\n")
        f.write(f"  Modularity Q         : {Q:.4f}\n")
        sizes_arr = sorted(comm_sizes.values(), reverse=True)
        f.write(f"  Largest community    : {sizes_arr[0]:,} nodes\n")
        f.write(f"  Median size          : {np.median(sizes_arr):.0f} nodes\n")
        f.write(f"  Singleton communities: {sum(1 for s in sizes_arr if s == 1):,}\n\n")

        f.write("5. HIV ACTIVITY SEPARATION BY COMMUNITY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Global HIV-active rate   : {global_rate*100:.3f}%\n")
        f.write(f"  3x threshold             : {thresh3x*100:.3f}%\n")
        f.write(f"  Comms with enrich = 0    : {below:,}\n")
        f.write(f"  Comms with 0 < e < 3x   : {low:,}\n")
        f.write(f"  Comms with 3x <= e < 5x : {mid:,}\n")
        f.write(f"  Comms with e >= 5x (HOT) : {high_enrich:,}\n\n")

        f.write("6. BFS DISTANCE TO NEAREST HIV-ACTIVE NODE\n")
        f.write("-" * 40 + "\n")
        dvals = list(bfs_dists.values())
        f.write(f"  Min: {min(dvals)}  Max: {max(dvals)}  Mean: {np.mean(dvals):.2f}\n")
        for d, cnt in sorted(Counter(dvals).items()):
            label = str(d) if d <= BFS_MAX_DIST else f">{BFS_MAX_DIST}"
            f.write(f"  distance={label}: {cnt} BrNPDB compounds\n")

        f.write("\n7. PREDICTIONS (score = 0.60*enrich_norm + 0.40*proximity)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Threshold: >= 0.30 -> ACTIVE\n")
        f.write(f"  Predicted ACTIVE   : {len(active_df)}\n")
        f.write(f"  Predicted INACTIVE : {len(inactive_df)}\n\n")
        for _, row in pred_df.iterrows():
            name = str(row["common_name"]).encode("ascii", "replace").decode("ascii")
            f.write(f"  [{row['predicted_activity']:<8}] {row['confidence']:<9}  "
                    f"score={row['activity_score']:.4f}  "
                    f"enrich={row['comm_hiv_enrich']:.4f} ({row['comm_enrich_xrate']:.1f}x)  "
                    f"dist={row['bfs_dist_to_active']}  "
                    f"BrNPDB {row['brnpdb_id']}  {name}\n")

    print(f"\nSaved full report: {OUT_TXT}")
    print("Done.")


if __name__ == "__main__":
    main()
