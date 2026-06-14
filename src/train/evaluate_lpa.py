import torch
from src.models.lpa import lpa_weighted
from src.utils.community import community_enrichment,community_summary
import pandas as pd
import matplotlib.pyplot as plt
import os

data = torch.load("results/similiarity_graph_hiv/undirected_graph_hiv.pt")

labels = lpa_weighted(
    edge_index=data.edge_index,
    edge_weight=data.edge_attr.squeeze(),
    num_nodes=data.num_nodes
)

community_members = []

for node_id in range(data.num_nodes):

    dataset = (
        "HIV"
        if data.node_dataset[node_id] == 0
        else "BrNPDB"
    )

    community_members.append({
        "node_id": node_id,
        "community": int(labels[node_id]),

        "dataset": dataset,

        "hiv_active":
            float(data.hiv_active[node_id]),

        "brnpdb_id":
            data.brnpdb_id[node_id]
            if hasattr(data, "brnpdb_id")
            else -1,

        "common_name":
            data.common_name[node_id]
            if hasattr(data, "common_name")
            else ""
    })

community_members_df = pd.DataFrame(
    community_members
)



community_summary(labels)

df = community_enrichment(
    labels,
    data.hiv_active,
    data.node_dataset
)

# Full results
print("Top communities (unfiltered):")
print(df.head(10))


# -------------------------------------------------------
# 🔹 Filter meaningful communities
# -------------------------------------------------------

df_filtered = df[
    (df["size"] >= 20) &          # remove tiny clusters
    (df["br_compounds"] > 0)      # keep only communities with BR compounds
]

print("\nTop meaningful communities:")
print(df_filtered.sort_values(by="enrichment", ascending=False).head(10))

brnpdb_communities = community_members_df[
    community_members_df["dataset"] == "BrNPDB"
]


community_compounds = (
    brnpdb_communities
    .groupby("community")
    .agg({
        "brnpdb_id": list,
        "common_name": list
    })
    .reset_index()
)

#--------------------------------------------------------------------------
# Save all results

df = community_enrichment(
    labels,
    data.hiv_active,
    data.node_dataset
)

OUTPUT_DIR = "results/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lpa_communities_all.csv"
    ),
    index=False
)

df_filtered.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lpa_communities_filtered.csv"
    ),
    index=False
)

community_members_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lpa_community_members.csv"
    ),
    index=False
)

brnpdb_communities.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lpa_brnpdb_compounds.csv"
    ),
    index=False
)

community_compounds.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lpa_community_compounds.csv"
    ),
    index=False
)

#-------------------------------------------------
#Community size distribution 

FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.figure(figsize=(8,5))

plt.hist(
    df["size"],
    bins=50
)

plt.xlabel("Community size")
plt.ylabel("Count")
plt.title("Distribution of LPA community sizes")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIG_DIR,
        "lpa_size_distribution.png"
    ),
    dpi=300
)

#---------------------------------------------------------
# Top enriched communities

top_comm = (
    df_filtered
    .sort_values("enrichment", ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))

plt.bar(
    top_comm["community"].astype(str),
    top_comm["enrichment"]
)

plt.xticks(rotation=45)

plt.ylabel("Enrichment")
plt.xlabel("Community ID")

plt.title(
    "Top HIV-enriched communities identified by LPA"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIG_DIR,
        "lpa_top_enriched_communities.png"
    ),
    dpi=300
)

#--------------------------------------------------
# Brazilian compounds per community

top_br = (
    df_filtered
    .sort_values("br_compounds", ascending=False)
    .head(15)
)

plt.figure(figsize=(10,5))

plt.bar(
    top_br["community"].astype(str),
    top_br["br_compounds"]
)

plt.xticks(rotation=45)

plt.ylabel("Brazilian compounds")
plt.xlabel("Community ID")

plt.title(
    "Brazilian compounds inside LPA communities"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIG_DIR,
        "lpa_brazilian_compounds.png"
    ),
    dpi=300
)

#--------------------------------------------------------
# Export report summary table

summary = {
    "num_nodes": data.num_nodes,
    "num_communities": len(torch.unique(labels)),
    "largest_community": int(df["size"].max()),
    "mean_community_size": float(df["size"].mean()),
    "communities_with_br": int((df["br_compounds"] > 0).sum()),
    "max_enrichment": float(df["enrichment"].max())
}


summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lpa_summary.csv"
    ),
    index=False
)

print(summary_df)