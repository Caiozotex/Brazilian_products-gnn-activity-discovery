import torch
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os

from src.utils.metrics_mlp import compute_metrics
from src.models.mlp_classifier import MLPClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

data = torch.load(
    "results/similiarity_graph_hiv/undirected_graph_hiv.pt"
)

embeddings = data.x
labels = data.hiv_active

# print(data)
# print(data.x.shape)
# print(data.hiv_active.shape)

# Keep only HIV dataset nodes
labeled_mask = labels != -1

X = embeddings[labeled_mask]
y = labels[labeled_mask]



active_mask = data.hiv_active == 1

active_nodes = active_mask.nonzero().squeeze()

print(len(active_nodes))

# print("Embeddings shape:", X.shape)
# print("Labels shape:", y.shape)

# print("Active:", (y == 1).sum())
# print("Inactive:", (y == 0).sum())

# print("All nodes:", len(labels))
# print("HIV nodes:", (labels != -1).sum().item())
# print("Brazilian nodes:", (labels == -1).sum().item())

#----------------------------------------------------------------


X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

# print("Train active:", (y_train == 1).sum())
# print("Train inactive:", (y_train == 0).sum())

# print("Val active:", (y_val == 1).sum())
# print("Val inactive:", (y_val == 0).sum())

# print("Test active:", (y_test == 1).sum())
# print("Test inactive:", (y_test == 0).sum())

# print(X.shape)
# print(X.std())
# print(X.mean())


# print("labels:",torch.unique(y, return_counts=True))

batch_size = 512

train_dataset = TensorDataset(
    X_train.float(),
    y_train.float()
)

val_dataset = TensorDataset(
    X_val.float(),
    y_val.float()
)

test_dataset = TensorDataset(
    X_test.float(),
    y_test.float()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

#---------------------------------------------------------------------------------------

active = (y == 1).sum()
inactive = (y == 0).sum()

pos_weight = (
    inactive.float() /
    active.float()
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [pos_weight],
        device=device
    )
)

input_dim = X.shape[1]

model = MLPClassifier(
    input_dim=input_dim,
    hidden_dim=128,
    dropout=0.3
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)

#------------------------------------------------------------------------

def evaluate(model, loader, criterion, device):

    model.eval()

    losses = []

    probs_all = []
    labels_all = []

    with torch.no_grad():

        for x_batch, y_batch in loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)

            loss = criterion(logits, y_batch)

            losses.append(loss.item())

            probs = torch.sigmoid(logits)

            probs_all.extend(
                probs.cpu().numpy()
            )

            labels_all.extend(
                y_batch.cpu().numpy()
            )

    probs_all = np.array(probs_all)
    labels_all = np.array(labels_all)

    metrics = compute_metrics(
        labels_all,
        probs_all,
        threshold=0.5
    )

    print("Prob mean:", probs_all.mean())
    print("Prob min:", probs_all.min())
    print("Prob max:", probs_all.max())

    print("Positive predictions:",(probs_all >= 0.5).sum())

    print("Total predictions:",len(probs_all))

    return (
        np.mean(losses),
        metrics
    )

#-------------------------------------------------------------------------------------------------------------

active_mean = X[y == 1].mean(dim=0)
inactive_mean = X[y == 0].mean(dim=0)

distance = torch.norm(
    active_mean - inactive_mean
)

print(
    torch.norm(active_mean),
    torch.norm(inactive_mean)
)

print(distance)

















# epochs = 200

# best_auc = 0

# patience = 20
# counter = 0


# best_model_path = "results/models"
# os.makedirs(best_model_path,exist_ok=True)

# for epoch in range(epochs):

#     model.train()

#     epoch_loss = 0

#     for x_batch, y_batch in train_loader:

#         x_batch = x_batch.to(device)
#         y_batch = y_batch.to(device)

#         optimizer.zero_grad()

#         logits = model(x_batch)

#         loss = criterion(
#             logits,
#             y_batch
#         )

#         loss.backward()

#         optimizer.step()

#         epoch_loss += loss.item()


#     val_loss, val_metrics = evaluate(
#         model,
#         val_loader,
#         criterion,
#         device
#     )

#     val_auc = val_metrics["roc_auc"]


#     print(
#         f"Epoch {epoch:03d} | "
#         f"Val Loss {val_loss:.4f} | "
#         f"AUC {val_metrics['roc_auc']:.4f} | "
#         f"Precision {val_metrics['precision']:.4f} | "
#         f"Recall {val_metrics['recall']:.4f} | "
#         f"F1 {val_metrics['f1']:.4f}"
#     )

#     print("Epoch Loss:", epoch_loss)

#     if val_auc > best_auc:

#         best_auc = val_auc

#         torch.save(
#             model.state_dict(),
#             os.path.join(best_model_path,"best_mlp.pt")
#         )

#         counter = 0

#     else:

#         counter += 1

#     if counter >= patience:

#         print("Early stopping")
#         break

#------------------------------------------------------------------------------------------

# model.load_state_dict(
#     torch.load(best_model_path)
# )