import pandas as pd
import torch

def community_enrichment(labels, hiv_active, node_dataset):
    """
    labels: [N] community ids
    hiv_active: [-1, 0, 1]
    node_dataset: 0 (HIV), 1 (BrNPDB)
    """

    df = pd.DataFrame({
        "community": labels.cpu().numpy(),
        "hiv_active": hiv_active.cpu().numpy(),
        "dataset": node_dataset.cpu().numpy()
    })

    results = []

    global_active = (df["hiv_active"] == 1).sum()
    global_total = (df["hiv_active"] != -1).sum()
    global_rate = global_active / max(global_total, 1)

    for comm_id, group in df.groupby("community"):
        size = len(group)

        hiv_labeled = group[group["hiv_active"] != -1]
        active = (hiv_labeled["hiv_active"] == 1).sum()
        inactive = (hiv_labeled["hiv_active"] == 0).sum()

        br_count = (group["dataset"] == 1).sum()

        frac_active = active / max((active + inactive), 1)

        enrichment = frac_active / global_rate if global_rate > 0 else 0

        results.append({
            "community": comm_id,
            "size": size,
            "active": active,
            "inactive": inactive,
            "br_compounds": br_count,
            "frac_active": frac_active,
            "enrichment": enrichment
        })

    result_df = pd.DataFrame(results)

    return result_df.sort_values(by="enrichment", ascending=False)

def community_summary(labels):
    unique, counts = torch.unique(labels, return_counts=True)

    print("Number of communities:", len(unique))
    print("Largest communities:", torch.topk(counts, k=10).values.tolist())