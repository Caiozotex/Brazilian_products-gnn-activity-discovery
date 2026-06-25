"""
Script 2: Para cada comunidade com BrNPDB, calcula f(i) = hiv_active_i / (hiv_active_i + hiv_inactive_i)
e compara com f(total) do dataset HIV completo.

Se f(i) >= f(total) * THRESHOLD, a comunidade é classificada como "active"
e todos os BrNPDB dela são marcados como candidatos ativos.

Threshold = 2.0 (enriquecimento 2× acima da taxa global)
  - Rationale: exige que a comunidade tenha o dobro da taxa basal de ativos,
    filtrando ruído de comunidades com apenas 1 ativo em dezenas de inativos.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
import networkx as nx
import pandas as pd

GEXF_PATH   = Path("results/similiarity_graph_hiv/hiv_knn_graph/graph_hiv_communities.gexf")
STATS_CSV   = Path("results/tables/brnpdb_community_stats.csv")
OUT_CSV     = Path("results/tables/brnpdb_community_classification.csv")

# Enriquecimento mínimo em relação à taxa global para classificar como ativa
THRESHOLD = 2.0

def compute_f_total(G: nx.Graph) -> tuple[int, int, float]:
    """Conta todos os nós HIV (com label) e retorna (n_active, n_total, f_total)."""
    n_active = 0
    n_hiv = 0
    for _, data in G.nodes(data=True):
        # Nós BrNPDB possuem brnpdb_id; ignorar
        if "brnpdb_id" in data and data["brnpdb_id"] not in (None, ""):
            continue
        hiv_val = data.get("hiv_active", -1.0)
        try:
            hiv_val = float(hiv_val)
        except (TypeError, ValueError):
            hiv_val = -1.0
        if hiv_val == 1.0:
            n_active += 1
            n_hiv += 1
        elif hiv_val == 0.0:
            n_hiv += 1
    f_total = n_active / n_hiv if n_hiv > 0 else 0.0
    return n_active, n_hiv, f_total

def main():
    print(f"Lendo {GEXF_PATH} ...")
    G = nx.read_gexf(str(GEXF_PATH))

    n_active_global, n_hiv_total, F_TOTAL = compute_f_total(G)
    print(f"\nDataset HIV completo:")
    print(f"  Total HIV (labeled): {n_hiv_total}")
    print(f"  HIV ativos:          {n_active_global}")
    print(f"  f(total):            {F_TOTAL:.4f}  ({F_TOTAL*100:.2f}%)")

    cutoff = F_TOTAL * THRESHOLD
    print(f"\nThreshold multiplicador: {THRESHOLD}×")
    print(f"Cutoff f(i) >=           {cutoff:.4f}  ({cutoff*100:.2f}%)")

    # Lê CSV do Script 1
    df = pd.read_csv(STATS_CSV)

    # Calcula f(i) e classifica
    def f_i(row):
        total_hiv = row["hiv_active"] + row["hiv_inactive"]
        if total_hiv == 0:
            return None  # sem nós HIV na comunidade — não classificável
        return row["hiv_active"] / total_hiv

    df["f_i"] = df.apply(f_i, axis=1)
    df["f_total"] = F_TOTAL
    df["threshold"] = THRESHOLD
    df["cutoff"] = cutoff

    def classify(row):
        if row["f_i"] is None:
            return "unclassifiable"
        return "active" if row["f_i"] >= cutoff else "inactive"

    df["community_class"] = df.apply(classify, axis=1)
    df["brnpdb_prediction"] = df["community_class"].map(
        {"active": "HIV_ACTIVE", "inactive": "HIV_INACTIVE", "unclassifiable": "UNKNOWN"}
    )

    df = df.sort_values(["community_class", "f_i"], ascending=[True, False])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # Resumo
    n_active_comm   = (df["community_class"] == "active").sum()
    n_inactive_comm = (df["community_class"] == "inactive").sum()
    n_uncl          = (df["community_class"] == "unclassifiable").sum()
    brnpdb_active   = df.loc[df["community_class"] == "active", "n_brnpdb"].sum()
    brnpdb_inactive = df.loc[df["community_class"] == "inactive", "n_brnpdb"].sum()
    brnpdb_uncl     = df.loc[df["community_class"] == "unclassifiable", "n_brnpdb"].sum()

    print(f"\n{'='*65}")
    print(f"{'Comunidades ativas':30s}: {n_active_comm:>4}  ({brnpdb_active} BrNPDB → HIV_ACTIVE)")
    print(f"{'Comunidades inativas':30s}: {n_inactive_comm:>4}  ({brnpdb_inactive} BrNPDB → HIV_INACTIVE)")
    print(f"{'Comunidades inclassificáveis':30s}: {n_uncl:>4}  ({brnpdb_uncl} BrNPDB → UNKNOWN)")
    print(f"{'='*65}")

    print(f"\nComunidades ATIVAS (f(i) >= {cutoff*100:.1f}%):")
    active_df = df[df["community_class"] == "active"][
        ["community_id", "n_brnpdb", "total_vertices", "hiv_active", "hiv_inactive", "f_i"]
    ].copy()
    active_df["f_i_pct"] = active_df["f_i"].map(lambda x: f"{x*100:.1f}%" if x is not None else "n/a")
    print(active_df.drop(columns="f_i").to_string(index=False))

    print(f"\nResultado salvo em: {OUT_CSV}")

if __name__ == "__main__":
    main()
