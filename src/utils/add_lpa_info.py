import pandas as pd
import torch

data = torch.load(
    "results/similiarity_graph_hiv/hiv_knn_graph/graph_hiv_communities.pt"
)

members = pd.read_csv(
    "results/tables/lpa_community_members.csv"
)

community = torch.full(
    (data.num_nodes,),
    -1,
    dtype=torch.long
)

for _, row in members.iterrows():
    community[int(row["node_id"])] = int(
        row["community"]
    )

data.community = community

torch.save(
    data,
    "results/similiarity_graph_hiv/hiv_knn_graph/graph_hiv_communities.pt"
)

data = torch.load(
    "results/similiarity_graph_hiv/hiv_knn_graph/graph_hiv_communities.pt"
)

print(data.community.shape)