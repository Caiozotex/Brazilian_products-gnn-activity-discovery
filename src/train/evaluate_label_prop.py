from src.models.label_propagation import label_propagation, get_unlabeled_predictions, compute_edge_homophily
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt


data = torch.load("results/similiarity_graph_hiv/undirected_graph_hiv.pt")

F = label_propagation(
    edge_index=data.edge_index,
    num_nodes=data.num_nodes,
    hiv_active=data.hiv_active,
    edge_weight=data.edge_attr.squeeze() if hasattr(data, "edge_attr") else None,
    alpha=0.7,
    device="cuda" if torch.cuda.is_available() else "cpu"
)


probs = get_unlabeled_predictions(F, data.hiv_active)

print("Predictions shape:", probs.shape)

print("Unlabeled predictions:", probs[:10])


# topk = torch.topk(probs, k=30)
# print("top predicted antiviral candidates:", topk.values)

homophily = compute_edge_homophily(data.edge_index, data.hiv_active)


#------------------------------------------------------------------
# Candidate ranking table

unlabeled_idx = torch.where(data.hiv_active == -1)[0]


# top_nodes = unlabeled_idx[topk.indices]


# -----------------------------------------------------
# All BrNPDB compounds
# -----------------------------------------------------

unlabeled_idx = torch.where(
    data.hiv_active == -1
)[0]

candidate_df = pd.DataFrame({
    "node_id":
        unlabeled_idx.cpu().numpy(),

    "brnpdb_id":
        data.brnpdb_id[unlabeled_idx].cpu().numpy(),

    "common_name":
        [data.common_name[i] for i in unlabeled_idx.cpu().numpy()],

    "hiv_activity_score":
        probs.cpu().numpy()
})

candidate_df = candidate_df.sort_values(
    by="hiv_activity_score",
    ascending=False
).reset_index(drop=True)

candidate_df.insert(
    0,
    "rank",
    range(1, len(candidate_df) + 1)
)

print("\nTop propagated candidates")
print(candidate_df.head(20))


OUTPUT_DIR = "results/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Creates 'results/tables'


candidate_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "label_propagation_screening.csv"
    ),
    index=False
)

# -----------------------------------------------------
# Distribution of propagated scores

plt.figure(figsize=(8,5))
plt.hist(probs.cpu().numpy(), bins=30)

plt.xlabel("Predicted HIV activity score")
plt.ylabel("Number of Brazilian compounds")
plt.title("Distribution of propagated HIV activity scores")

plt.tight_layout()

DIR_OUTPUT_figs = "results/figures"
os.makedirs(DIR_OUTPUT_figs, exist_ok=True)

plt.savefig(
        os.path.join(
        DIR_OUTPUT_figs,
        "label_prop_score_distribution.png"
    ),
    dpi=300
)

# top_scores = topk.values.cpu().numpy()

top_scores = (
    candidate_df["hiv_activity_score"]
    .head(30)
    .values
)

plt.figure(figsize=(10,5))
plt.bar(range(len(top_scores)), top_scores)

plt.xlabel("Candidate rank")
plt.ylabel("Propagation score")
plt.title("Top predicted antiviral candidates")

plt.tight_layout()

plt.savefig(
    os.path.join(
        DIR_OUTPUT_figs,
        "top_antiviral_candidates.png"
    ),
    dpi=300
)

#---------------------------------------------------------

summary_df = pd.DataFrame({
    "Metric": [
        "Nodes",
        "Edges",
        "Known HIV active",
        "Known HIV inactive",
        "Unlabeled compounds",
        "Edge homophily",
        "Predicted active (>0.5)",
        "Mean propagation score",
        "Median propagation score",
        "Maximum propagation score"
    ],
    "Value": [
        data.num_nodes,
        data.edge_index.shape[1],
        int((data.hiv_active == 1).sum()),
        int((data.hiv_active == 0).sum()),
        int((data.hiv_active == -1).sum()),
        float(homophily),
        int((probs > 0.5).sum()),
        float(probs.mean()),
        float(probs.median()),
        float(probs.max())
    ]
})

summary_df.to_csv(
        os.path.join(
        OUTPUT_DIR,
        "label_prop_summary.csv"
    ),
    index=False
)
