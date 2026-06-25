"""
Script 1 (Louvain 27): Para cada uma das 27 comunidades Louvain que contém
ao menos um composto BrNPDB, extrai:
  community_id, n_brnpdb, total_vertices, hiv_active, hiv_inactive.

Fonte: results/tables/louvain_node_communities.csv
  (colunas: node_id, community, is_brnpdb, is_active)

Saída: results/tables/brnpdb_louvain27_stats.csv
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
import pandas as pd

LOUVAIN_CSV = Path("results/tables/louvain_node_communities.csv")
OUT_CSV     = Path("results/tables/brnpdb_louvain27_stats.csv")

def main():
    df = pd.read_csv(LOUVAIN_CSV)
    print(f"Nós carregados: {len(df)}  |  Comunidades: {df['community'].nunique()}")

    brnpdb = df[df["is_brnpdb"] == True]
    hiv    = df[df["is_brnpdb"] == False]

    n_brnpdb_per_comm  = brnpdb.groupby("community").size().rename("n_brnpdb")
    total_per_comm     = df.groupby("community").size().rename("total_vertices")
    hiv_active_per     = hiv[hiv["is_active"] == True].groupby("community").size().rename("hiv_active")
    hiv_inactive_per   = hiv[hiv["is_active"] == False].groupby("community").size().rename("hiv_inactive")

    stats = (n_brnpdb_per_comm
             .to_frame()
             .join(total_per_comm, how="left")
             .join(hiv_active_per, how="left")
             .join(hiv_inactive_per, how="left")
             .fillna(0)
             .astype(int)
             .reset_index()
             .rename(columns={"community": "community_id"}))

    # Filtra comunidades sem BrNPDB
    stats = stats[stats["n_brnpdb"] > 0].sort_values("n_brnpdb", ascending=False)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUT_CSV, index=False)

    print(f"\nComunidades com BrNPDB: {len(stats)}")
    print(stats.to_string(index=False))
    print(f"\nSalvo em: {OUT_CSV}")

if __name__ == "__main__":
    main()
