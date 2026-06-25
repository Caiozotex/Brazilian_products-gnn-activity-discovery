"""
Community meta-graph builder.

Reads a kNN GEXF and a community assignment (from GEXF node attributes OR
from an external CSV), then builds a coarser graph where:
  - each node  = one Louvain community
  - edge weight = number of inter-community edges in the original graph
  - intra_edges  = number of intra-community edges (stored as node attribute)

Node attributes exported:
  community_id   (original community ID)
  label          (chemical family name, if available)
  size           (number of nodes)
  n_active / n_inactive
  hiv_rate       (fraction HIV-active)
  intra_edges

Usage
-----
# Mode A – communities already in the GEXF node attributes:
  python src/analysis/community_meta_graph.py \\
      results/similiarity_graph_hiv/hiv_knn_graph/graph_hiv_communities.gexf \\
      results/similiarity_graph_hiv/hiv_knn_graph/community_meta_graph_2550.gexf

# Mode B – supply an external CSV (node_id, community columns):
  python src/analysis/community_meta_graph.py \\
      results/similiarity_graph_hiv/hiv_knn_graph.gexf \\
      results/similiarity_graph_hiv/hiv_knn_graph/community_meta_graph_27.gexf \\
      --communities-csv results/tables/louvain_node_communities.csv \\
      --labels-csv      results/tables/community_chemical_families.csv
"""

import os, sys, time, argparse
import networkx as nx
import pandas as pd
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("in_gexf",  nargs="?",
                    default="results/similiarity_graph_hiv/hiv_knn_graph/graph_hiv_communities.gexf")
parser.add_argument("out_gexf", nargs="?", default=None)
parser.add_argument("--communities-csv", default=None,
                    help="CSV with columns node_id and community (overrides GEXF attributes)")
parser.add_argument("--labels-csv", default=None,
                    help="CSV with columns comunidade and familia_quimica")
args = parser.parse_args()

IN_GEXF = args.in_gexf
if args.out_gexf:
    OUT_GEXF = args.out_gexf
else:
    stem = os.path.splitext(os.path.basename(IN_GEXF))[0]
    OUT_GEXF = os.path.join(os.path.dirname(IN_GEXF), f"{stem}_meta.gexf")

os.makedirs(os.path.dirname(OUT_GEXF) or ".", exist_ok=True)

print(f"Input  : {IN_GEXF}")
print(f"Output : {OUT_GEXF}")
if args.communities_csv:
    print(f"Comms  : {args.communities_csv}")
if args.labels_csv:
    print(f"Labels : {args.labels_csv}")


# ── 1. Load graph ─────────────────────────────────────────────────────────────

print("\nLoading graph ...")
t0 = time.time()
G  = nx.read_gexf(IN_GEXF)
print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges  ({time.time()-t0:.1f}s)")


# ── 2. Community assignments ──────────────────────────────────────────────────

node_community = {}   # node_str -> int community id
node_active    = {}   # node_str -> bool

if args.communities_csv:
    # External CSV: node_id (int) -> community (int)
    comm_df = pd.read_csv(args.communities_csv)
    id_to_comm = dict(zip(comm_df["node_id"].astype(str), comm_df["community"].astype(int)))

    # Activity from CSV if present, else from GEXF
    if "is_active" in comm_df.columns:
        id_to_active = dict(zip(comm_df["node_id"].astype(str), comm_df["is_active"].astype(bool)))
    else:
        id_to_active = {}

    for n, data in G.nodes(data=True):
        node_community[n] = id_to_comm.get(str(int(float(n))), -1)
        if id_to_active:
            node_active[n] = id_to_active.get(str(int(float(n))), False)
        else:
            active = data.get("hiv_active", 0)
            node_active[n] = bool(float(active)) if active is not None else False
else:
    # Communities stored in the GEXF node attribute "community"
    for n, data in G.nodes(data=True):
        comm = data.get("community", data.get("Modularity Class", -1))
        node_community[n] = int(float(comm)) if comm is not None else -1
        active = data.get("hiv_active", 0)
        node_active[n] = bool(float(active)) if active is not None else False

raw_ids    = sorted(set(node_community.values()))
K          = len(raw_ids)
raw_to_seq = {raw: seq for seq, raw in enumerate(raw_ids)}
print(f"  {K:,} unique communities")


# ── 3. Optional chemical-family labels ───────────────────────────────────────

comm_label = {}
if args.labels_csv:
    lab_df = pd.read_csv(args.labels_csv)
    for _, row in lab_df.iterrows():
        comm_label[int(row["comunidade"])] = str(row["familia_quimica"])


# ── 4. Per-community statistics ───────────────────────────────────────────────

comm_size     = defaultdict(int)
comm_n_active = defaultdict(int)

for n, raw in node_community.items():
    c = raw_to_seq[raw]
    comm_size[c] += 1
    if node_active[n]:
        comm_n_active[c] += 1


# ── 5. Count inter-community edges ────────────────────────────────────────────

print("Counting inter-community edges ...")
t1 = time.time()
edge_counts = defaultdict(int)

for u, v in G.edges():
    cu = raw_to_seq[node_community[u]]
    cv = raw_to_seq[node_community[v]]
    edge_counts[(min(cu, cv), max(cu, cv))] += 1

n_inter = sum(1 for (a, b) in edge_counts if a != b)
n_intra = sum(1 for (a, b) in edge_counts if a == b)
print(f"  Done in {time.time()-t1:.1f}s — {n_intra} intra, {n_inter} inter community-pairs")

max_pairs = K * (K - 1) // 2
density   = n_inter / max_pairs if max_pairs > 0 else 0
print(f"  Meta-graph density: {density:.4f}  ({n_inter:,} / {max_pairs:,} possible pairs)")
if density > 0.9:
    print("  WARNING: meta-graph near-complete — may not be informative.")


# ── 6. Build meta-graph ───────────────────────────────────────────────────────

M = nx.Graph()

for seq in range(K):
    raw   = raw_ids[seq]
    size  = comm_size[seq]
    n_act = comm_n_active[seq]
    rate  = n_act / size if size > 0 else 0.0
    label = comm_label.get(raw, f"C{raw}")
    M.add_node(
        str(seq),
        label=label,
        community_id=raw,
        size=size,
        n_active=n_act,
        n_inactive=size - n_act,
        hiv_rate=round(rate, 4),
        intra_edges=edge_counts.get((seq, seq), 0),
    )

for (ca, cb), count in edge_counts.items():
    if ca != cb:
        M.add_edge(str(ca), str(cb), weight=count)

print(f"\nMeta-graph: {M.number_of_nodes()} nodes, {M.number_of_edges()} edges")


# ── 7. Print summary ──────────────────────────────────────────────────────────

sizes_list = sorted((comm_size[c] for c in range(K)), reverse=True)
print(f"\nCommunity size — largest: {sizes_list[0]:,}  median: {sizes_list[K//2]:,}  smallest: {sizes_list[-1]:,}")
print(f"Singletons (size=1): {sum(1 for s in sizes_list if s == 1)}")

print(f"\n{'Seq':>4}  {'CommID':>7}  {'Size':>6}  {'Active':>6}  {'HIV%':>6}  {'Intra':>7}  Label")
print("-" * 100)
for seq in sorted(range(K), key=lambda c: comm_size[c], reverse=True)[:30]:
    raw   = raw_ids[seq]
    size  = comm_size[seq]
    n_act = comm_n_active[seq]
    intra = edge_counts.get((seq, seq), 0)
    label = comm_label.get(raw, f"C{raw}")
    print(f"{seq:>4}  {raw:>7}  {size:>6,}  {n_act:>6,}  {n_act/size*100:>5.1f}%  {intra:>7,}  {label[:60]}")

print("\nTop 15 inter-community edges:")
for u, v, d in sorted(M.edges(data=True), key=lambda e: e[2]["weight"], reverse=True)[:15]:
    lu = M.nodes[u]["label"][:30]
    lv = M.nodes[v]["label"][:30]
    print(f"  C{u:>4} <-> C{v:>4}  w={d['weight']:>6,}   {lu}  |  {lv}")


# ── 8. Export ─────────────────────────────────────────────────────────────────

nx.write_gexf(M, OUT_GEXF)
print(f"\nSaved: {OUT_GEXF}")
