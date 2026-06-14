import torch
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from src.models.gin_model import GINEEncoder
from src.models.task_models import HIVModel

from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import os


# --------------------------------------------------
# Load model
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

encoder = GINEEncoder(
    in_channels=35,
    edge_dim=9,
    hidden_channels=128,
    num_layers=4,
    dropout=0.2
)

model = HIVModel(
    encoder=encoder,
    hidden_dim=128
)


checkpoint_path = (
    "models/checkpoints/hiv/best_model.pt"
)


model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)

model.to(device)
model.eval()

#----------------------------------------------------

# Directories where individual .pt files live
HIV_DIR = "data/processed/graphs_hiv"
ANTIVIRAL_DIR = "data/processed/graphs_brnpdb_antiviral"

# print(f"NuBBE (AntioxidantRos) graphs: {len(nubbe_list)}")

# HIV dataset (binary classification: 1 = active, 0 = inactive)
hiv_list = []
for fname in sorted(os.listdir(HIV_DIR)):
    if fname.endswith('.pt'):
        data = torch.load(os.path.join(HIV_DIR, fname))
        # Ensure y is a scalar tensor (binary label)
        # HIVDataset stores binary label in data.y (shape (1,)) or data.hiv_active
        if hasattr(data, 'hiv_active'):
            data.y = data.hiv_active  # already a tensor
        elif hasattr(data, 'y') and data.y.numel() == 1:
            pass  # already single value
        elif hasattr(data, 'y') and data.y.numel() > 1:
            data.y = data.y[0]  # take first element if multi-label
        else:
            raise ValueError(f"HIV graph {fname} missing label")
        # Ensure shape (1,) for consistency
        data.y = data.y.view(1)
        hiv_list.append(data)

# BrNPDB Antiviral dataset (all compounds are antiviral, so label = 1.0)
antiviral_list = []
for fname in sorted(os.listdir(ANTIVIRAL_DIR)):
    if fname.endswith('.pt'):
        data = torch.load(os.path.join(ANTIVIRAL_DIR, fname))
        # Assign a fixed label (1 = antiviral) – useful for pretraining or as positive set
        data.y = torch.tensor([1.0])
        # Optional: store the original brnpdb_id or common_name if needed
        # if hasattr(data, 'brnpdb_id'): ...
        antiviral_list.append(data)

print(f"HIV graphs: {len(hiv_list)}")
print(f"BrNPDB Antiviral graphs: {len(antiviral_list)}")





hiv_loader = DataLoader(
    hiv_list,
    batch_size=64,
    shuffle=False
)
antiviral_loader = DataLoader(
    antiviral_list, 
    batch_size=64, 
    shuffle=False)

# sample = antiviral_list[0]

# print(sample)

#----------------------------------------------------------------



# --------------------------------------------------
# Evaluation
# --------------------------------------------------

all_probs = []
all_labels = []

model.eval()

with torch.no_grad():

    for batch in hiv_loader:

        batch = batch.to(device)

        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )

        probs = torch.sigmoid(
            logits.squeeze()
        )

        all_probs.extend(
            probs.cpu().numpy()
        )

        all_labels.extend(
            batch.y.cpu().numpy()
        )


# --------------------------------------------------
# Metrics
# --------------------------------------------------

roc_auc = roc_auc_score(
    all_labels,
    all_probs
)

pr_auc = average_precision_score(
    all_labels,
    all_probs
)

print("ROC-AUC:", roc_auc)
print("PR-AUC :", pr_auc)


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

import numpy as np

all_probs = np.array(all_probs)

print("Mean:", all_probs.mean())
print("Min :", all_probs.min())
print("Max :", all_probs.max())

from sklearn.metrics import confusion_matrix

preds = (all_probs >= 0.5)

tn, fp, fn, tp = confusion_matrix(
    all_labels,
    preds
).ravel()

print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print("TN:", tn)

precision = tp / (tp + fp)
recall = tp / (tp + fn)


from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(
    all_labels,
    all_probs
)

f1_scores = (
    2 * precision * recall /
    (precision + recall + 1e-8)
)

best_idx = f1_scores.argmax()

print("Best threshold:", thresholds[best_idx])
print("Best F1:", f1_scores[best_idx])
print("Precision:", precision[best_idx])
print("Recall:", recall[best_idx])


# print(
#     "Predicted active:",
#     (all_probs >= 0.5).sum()
# )

# --------------------------------------------------
# BrNPDB Screening
# --------------------------------------------------

brnpdb_probs = []
brnpdb_ids = []
brnpdb_names = []

model.eval()

with torch.no_grad():

    for batch in antiviral_loader:

        batch = batch.to(device)

        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )

        probs = torch.sigmoid(
            logits.squeeze()
        )

        brnpdb_probs.extend(
            probs.cpu().numpy()
        )

        # Recover metadata
        if hasattr(batch, "brnpdb_id"):
            brnpdb_ids.extend(
                batch.brnpdb_id.cpu().numpy().tolist()
            )

        if hasattr(batch, "common_name"):
            brnpdb_names.extend(
                list(batch.common_name)
            )

mean_score = np.mean(brnpdb_probs)

median_score = np.median(brnpdb_probs)

max_score = np.max(brnpdb_probs)

top10_mean = np.mean(
    np.sort(brnpdb_probs)[-10:]
)

print("Mean HIV probability:",mean_score )
print("Median HIV probability:",median_score )
print("Maximum HIV probability:",max_score)
print("Top-10 mean probability:",top10_mean)


brnpdb_map = []

for data in antiviral_list:

    brnpdb_map.append({
        "brnpdb_id": data.brnpdb_id,
        "common_name": data.common_name
    })

brnpdb_df = pd.DataFrame(brnpdb_map)



candidate_df = pd.DataFrame({
    "brnpdb_id": brnpdb_ids,
    "common_name": brnpdb_names,
    "predicted_hiv_probability": brnpdb_probs
})

candidate_df = candidate_df.sort_values(
    by="predicted_hiv_probability",
    ascending=False
).reset_index(drop=True)


candidate_df["rank"] = (
    candidate_df.index + 1
)


candidate_df = candidate_df[
    [
        "rank",
        "brnpdb_id",
        "common_name",
        "predicted_hiv_probability"
    ]
]

print("\nTop 20 BrNPDB candidates")

print(
    candidate_df.head(20)
)


OUTPUT_DIR = "results/tables"

candidate_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "brnpdb_hiv_screening.csv"
    ),
    index=False
)

print(
    "\nSaved:",
    "results/tables/brnpdb_hiv_screening.csv"
)