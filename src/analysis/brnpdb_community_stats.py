"""
Script 1: Para cada comunidade que contém ao menos um composto BrNPDB,
extrai: community_id, n_brnpdb, total_vertices, hiv_active, hiv_inactive.
Saída: results/tables/brnpdb_community_stats.csv
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
import networkx as nx
import pandas as pd
from collections import defaultdict

GEXF_PATH = Path("results/similiarity_graph_hiv/hiv_knn_graph/graph_hiv_communities.gexf")
OUT_CSV   = Path("results/tables/brnpdb_community_stats.csv")

def main():
    print(f"Lendo {GEXF_PATH} ...")
    G = nx.read_gexf(str(GEXF_PATH))
    print(f"  {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")

    # Agrega estatísticas por comunidade
    stats = defaultdict(lambda: {"n_brnpdb": 0, "total": 0, "hiv_active": 0, "hiv_inactive": 0})

    for _, data in G.nodes(data=True):
        comm = data.get("community")
        if comm is None:
            continue
        comm = int(comm)
        stats[comm]["total"] += 1

        has_brnpdb = "brnpdb_id" in data and data["brnpdb_id"] not in (None, "")
        if has_brnpdb:
            stats[comm]["n_brnpdb"] += 1
        else:
            # nó HIV: hiv_active é 0.0 ou 1.0
            hiv_val = data.get("hiv_active", -1.0)
            try:
                hiv_val = float(hiv_val)
            except (TypeError, ValueError):
                hiv_val = -1.0
            if hiv_val == 1.0:
                stats[comm]["hiv_active"] += 1
            elif hiv_val == 0.0:
                stats[comm]["hiv_inactive"] += 1

    # Filtra apenas comunidades com pelo menos 1 BrNPDB
    rows = []
    for comm_id, s in sorted(stats.items()):
        if s["n_brnpdb"] > 0:
            rows.append({
                "community_id":  comm_id,
                "n_brnpdb":      s["n_brnpdb"],
                "total_vertices": s["total"],
                "hiv_active":    s["hiv_active"],
                "hiv_inactive":  s["hiv_inactive"],
            })

    df = pd.DataFrame(rows).sort_values("n_brnpdb", ascending=False)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nComunidades com BrNPDB: {len(df)}")
    print(df.to_string(index=False))
    print(f"\nSalvo em: {OUT_CSV}")

if __name__ == "__main__":
    main()
