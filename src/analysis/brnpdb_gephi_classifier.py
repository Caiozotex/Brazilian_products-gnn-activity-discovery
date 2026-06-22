"""
Script 2 (Gephi Modularity): Calcula f(i) = hiv_active_i / (hiv_active_i + hiv_inactive_i)
para cada comunidade com BrNPDB e compara com f(total) do dataset HIV completo.

Se f(i) >= f(total) * THRESHOLD, a comunidade é "active" e todos os seus
BrNPDB são marcados como HIV_ACTIVE.

Threshold = 2.0  (enriquecimento 2× acima da taxa global)

Fonte stats : results/tables/brnpdb_gephi_stats.csv
Fonte global: results/tables/gephi_modularity.csv
Saída       : results/tables/brnpdb_gephi_classification.csv
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
import pandas as pd

GEPHI_CSV = Path("results/tables/gephi_modularity.csv")
STATS_CSV = Path("results/tables/brnpdb_gephi_stats.csv")
OUT_CSV   = Path("results/tables/brnpdb_gephi_classification.csv")

THRESHOLD = 2.0

def main():
    full = pd.read_csv(GEPHI_CSV)
    hiv  = full[full["6"].isna()]  # nós sem brnpdb_id → HIV

    n_active_global = int((hiv["2"] == 1).sum())
    n_hiv_total     = len(hiv)
    F_TOTAL         = n_active_global / n_hiv_total

    cutoff = F_TOTAL * THRESHOLD

    print(f"Dataset HIV completo:")
    print(f"  Total HIV (labeled): {n_hiv_total}")
    print(f"  HIV ativos:          {n_active_global}")
    print(f"  f(total):            {F_TOTAL:.4f}  ({F_TOTAL*100:.2f}%)")
    print(f"\nThreshold multiplicador: {THRESHOLD}×")
    print(f"Cutoff f(i) >=           {cutoff:.4f}  ({cutoff*100:.2f}%)")

    df = pd.read_csv(STATS_CSV)

    def f_i(row):
        total_hiv = row["hiv_active"] + row["hiv_inactive"]
        return row["hiv_active"] / total_hiv if total_hiv > 0 else None

    df["f_i"]       = df.apply(f_i, axis=1)
    df["f_total"]   = F_TOTAL
    df["threshold"] = THRESHOLD
    df["cutoff"]    = cutoff

    def classify(row):
        if row["f_i"] is None:
            return "unclassifiable"
        return "active" if row["f_i"] >= cutoff else "inactive"

    df["community_class"]   = df.apply(classify, axis=1)
    df["brnpdb_prediction"] = df["community_class"].map(
        {"active": "HIV_ACTIVE", "inactive": "HIV_INACTIVE", "unclassifiable": "UNKNOWN"}
    )

    df = df.sort_values(["community_class", "f_i"], ascending=[True, False])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    n_active_comm   = (df["community_class"] == "active").sum()
    n_inactive_comm = (df["community_class"] == "inactive").sum()
    n_uncl          = (df["community_class"] == "unclassifiable").sum()
    brnpdb_active   = df.loc[df["community_class"] == "active",          "n_brnpdb"].sum()
    brnpdb_inactive = df.loc[df["community_class"] == "inactive",        "n_brnpdb"].sum()
    brnpdb_uncl     = df.loc[df["community_class"] == "unclassifiable",  "n_brnpdb"].sum()

    print(f"\n{'='*65}")
    print(f"{'Comunidades ativas':30s}: {n_active_comm:>3}  ({brnpdb_active} BrNPDB → HIV_ACTIVE)")
    print(f"{'Comunidades inativas':30s}: {n_inactive_comm:>3}  ({brnpdb_inactive} BrNPDB → HIV_INACTIVE)")
    print(f"{'Comunidades inclassificáveis':30s}: {n_uncl:>3}  ({brnpdb_uncl} BrNPDB → UNKNOWN)")
    print(f"{'='*65}")

    active_df = df[df["community_class"] == "active"][
        ["community_id", "n_brnpdb", "total_vertices", "hiv_active", "hiv_inactive", "f_i"]
    ].copy()
    active_df["f_i_pct"] = active_df["f_i"].map(lambda x: f"{x*100:.1f}%" if x is not None else "n/a")

    print(f"\nComunidades ATIVAS (f(i) >= {cutoff*100:.1f}%):")
    print(active_df.drop(columns="f_i").to_string(index=False))

    print(f"\nResultado salvo em: {OUT_CSV}")

if __name__ == "__main__":
    main()
