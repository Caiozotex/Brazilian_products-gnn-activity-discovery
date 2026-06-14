import pandas as pd
from sklearn.preprocessing import MinMaxScaler

mlp_df = pd.read_csv(
    "results/tables/brnpdb_hiv_screening.csv"
)

lp_df = pd.read_csv(
    "results/tables/label_propagation_screening.csv"
)

community_df = pd.read_csv(
    "results/tables/lpa_community_members.csv"
)

mlp_df = mlp_df.rename(
    columns={
        "predicted_hiv_probability": "mlp_score",
        "rank": "mlp_rank"
    }
)

lp_df = lp_df.rename(
    columns={
        "hiv_activity_score": "lp_score",
        "rank": "lp_rank"
    }
)

community_df = community_df[
    community_df["dataset"] == "BrNPDB"
]

community_df = community_df[
    [
        "brnpdb_id",
        "community"
    ]
]

community_df["brnpdb_id"] = (
    community_df["brnpdb_id"]
    .astype(str)
    .str.extract(r"tensor\((\d+)\)")
)

community_df["brnpdb_id"] = pd.to_numeric(
    community_df["brnpdb_id"],
    errors="coerce"
)

community_df = community_df.dropna(
    subset=["brnpdb_id"]
)

community_df["brnpdb_id"] = (
    community_df["brnpdb_id"]
    .astype(int)
)


merged = pd.merge(
    mlp_df,
    lp_df[
        [
            "brnpdb_id",
            "lp_score",
            "lp_rank"
        ]
    ],
    on="brnpdb_id",
    how="outer"
)

merged = pd.merge(
    merged,
    community_df,
    on="brnpdb_id",
    how="left"
)

scaler = MinMaxScaler()

merged[["mlp_norm"]] = scaler.fit_transform(
    merged[["mlp_score"]]
)

merged[["lp_norm"]] = scaler.fit_transform(
    merged[["lp_score"]]
)

merged["consensus_score"] = (
    merged["mlp_norm"] +
    merged["lp_norm"]
) / 2

merged = merged.sort_values(
    "consensus_score",
    ascending=False
)

merged["consensus_rank"] = (
    range(
        1,
        len(merged) + 1
    )
)

community_stats = (
    merged
    .groupby("community")
    .agg({
        "consensus_score": "mean",
        "brnpdb_id": "count"
    })
)

merged = merged[
    [
        "consensus_rank",
        "brnpdb_id",
        "common_name",
        "community",
        "mlp_score",
        "lp_score",
        "consensus_score",
        "mlp_rank",
        "lp_rank"
    ]
]

merged.to_csv(
    "results/tables/brnpdb_consensus_screening.csv",
    index=False
)