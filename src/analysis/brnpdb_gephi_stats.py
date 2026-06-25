"""
Script 1 (Gephi Modularity): Para cada comunidade Gephi que contém ao menos
um composto BrNPDB, extrai:
  community_id, n_brnpdb, total_vertices, hiv_active, hiv_inactive.

Fonte : results/tables/gephi_modularity.csv
  Colunas relevantes:
    '1'              → dataset ("HIV" | "Antiviral")
    '2'              → hiv_active (0 | 1 | -1 para BrNPDB)
    '6'              → brnpdb_id (NaN para nós HIV)
    modularity_class → comunidade Gephi (0-21)

Saída: results/tables/brnpdb_gephi_stats.csv
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
import pandas as pd

GEPHI_CSV = Path("results/tables/gephi_modularity.csv")
OUT_CSV   = Path("results/tables/brnpdb_gephi_stats.csv")

def main():
    df = pd.read_csv(GEPHI_CSV)
    print(f"Nós carregados: {len(df)}  |  Comunidades Gephi: {df['modularity_class'].nunique()}")

    is_brnpdb = df["6"].notna()
    brnpdb    = df[is_brnpdb]
    hiv       = df[~is_brnpdb]

    n_brnpdb_per   = brnpdb.groupby("modularity_class").size().rename("n_brnpdb")
    total_per      = df.groupby("modularity_class").size().rename("total_vertices")
    hiv_active_per = hiv[hiv["2"] == 1].groupby("modularity_class").size().rename("hiv_active")
    hiv_inactive_per = hiv[hiv["2"] == 0].groupby("modularity_class").size().rename("hiv_inactive")

    stats = (n_brnpdb_per
             .to_frame()
             .join(total_per,       how="left")
             .join(hiv_active_per,  how="left")
             .join(hiv_inactive_per, how="left")
             .fillna(0)
             .astype(int)
             .reset_index()
             .rename(columns={"modularity_class": "community_id"}))

    stats = stats[stats["n_brnpdb"] > 0].sort_values("n_brnpdb", ascending=False)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUT_CSV, index=False)

    print(f"\nComunidades com BrNPDB: {len(stats)}")
    print(stats.to_string(index=False))
    print(f"\nSalvo em: {OUT_CSV}")

if __name__ == "__main__":
    main()
