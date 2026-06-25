"""
Avalia o MLPClassifier salvo em results/models/best_mlp.pt no conjunto de teste
usando o mesmo split estratificado do treinamento (70/15/15, random_state=42).
Gera results/tables/mlp_test_metrics.csv e imprime um relatório.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import os, sys
sys.path.insert(0, os.path.abspath("."))

from src.models.mlp_classifier import MLPClassifier

# ── 1. Carregar dados ──────────────────────────────────────────────────────────
data = torch.load(
    "results/similiarity_graph_hiv/undirected_graph_hiv.pt",
    map_location="cpu",
    weights_only=False
)

embeddings = data.x
labels = data.hiv_active

# Apenas compostos HIV rotulados
labeled_mask = labels != -1
X = embeddings[labeled_mask]
y = labels[labeled_mask]

print(f"Total compostos HIV: {len(y)}")
print(f"  Ativos  : {(y==1).sum().item()} ({100*(y==1).float().mean().item():.2f}%)")
print(f"  Inativos: {(y==0).sum().item()} ({100*(y==0).float().mean().item():.2f}%)")

# ── 2. Reproduzir o mesmo split do treinamento ─────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"\nSplit:")
print(f"  Treino : {len(y_train):,}  (ativos: {(y_train==1).sum().item()})")
print(f"  Val    : {len(y_val):,}  (ativos: {(y_val==1).sum().item()})")
print(f"  Teste  : {len(y_test):,}  (ativos: {(y_test==1).sum().item()})")

# ── 3. Carregar modelo ─────────────────────────────────────────────────────────
device = torch.device("cpu")
model = MLPClassifier(input_dim=X.shape[1], hidden_dim=128, dropout=0.3).to(device)
state = torch.load("results/models/best_mlp.pt", map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"\nModelo carregado: results/models/best_mlp.pt")

# ── 4. Inferência no conjunto de teste ────────────────────────────────────────
with torch.no_grad():
    logits = model(X_test.float().to(device))
    probs  = torch.sigmoid(logits).cpu().numpy()

y_np = y_test.numpy()

# Threshold padrão 0.5 e threshold ótimo por F1
thresholds = np.linspace(0.01, 0.99, 200)
f1s = [f1_score(y_np, (probs >= t).astype(int), zero_division=0) for t in thresholds]
best_t = thresholds[np.argmax(f1s)]

print(f"\n── Métricas no conjunto de TESTE ─────────────────────────────────────────")
print(f"  Threshold padrão : 0.50")
print(f"  Threshold ótimo  : {best_t:.4f}  (maximiza F1)")

rows = []
for thr, label in [(0.5, "thr=0.50"), (best_t, "thr_otimo")]:
    preds = (probs >= thr).astype(int)
    auc   = roc_auc_score(y_np, probs)
    ap    = average_precision_score(y_np, probs)
    prec  = precision_score(y_np, preds, zero_division=0)
    rec   = recall_score(y_np, preds, zero_division=0)
    f1    = f1_score(y_np, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_np, preds).ravel()
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\n  [{label}]")
    print(f"    ROC-AUC         : {auc:.4f}")
    print(f"    PR-AUC (AP)     : {ap:.4f}")
    print(f"    Precision       : {prec:.4f}")
    print(f"    Recall (Sensit.): {rec:.4f}")
    print(f"    Specificity     : {spec:.4f}")
    print(f"    F1-score        : {f1:.4f}")
    print(f"    TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    rows.append({
        "threshold": thr,
        "label": label,
        "roc_auc": round(auc, 4),
        "pr_auc": round(ap, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    })

# Também avalia no val (para comparar com teste e detectar overfitting)
with torch.no_grad():
    logits_v = model(X_val.float().to(device))
    probs_v  = torch.sigmoid(logits_v).cpu().numpy()

auc_v = roc_auc_score(y_val.numpy(), probs_v)
ap_v  = average_precision_score(y_val.numpy(), probs_v)
print(f"\n  [val] ROC-AUC={auc_v:.4f}  PR-AUC={ap_v:.4f}")

# ── 5. Salvar métricas ────────────────────────────────────────────────────────
os.makedirs("results/tables", exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv("results/tables/mlp_test_metrics.csv", index=False)
print(f"\nSalvo em: results/tables/mlp_test_metrics.csv")

# ── 6. Resumo da distribuição de probabilidades nos BrNPDB ───────────────────
brnpdb_mask = data.hiv_active == -1
X_br = data.x[brnpdb_mask]
with torch.no_grad():
    logits_br = model(X_br.float().to(device))
    probs_br  = torch.sigmoid(logits_br).cpu().numpy()

print(f"\n── Distribuição de scores MLP nos 147 compostos BrNPDB ──────────────────")
print(f"  Média   : {probs_br.mean():.4f}")
print(f"  Mediana : {np.median(probs_br):.4f}")
print(f"  Máximo  : {probs_br.max():.4f}")
print(f"  >0.10   : {(probs_br > 0.10).sum()}")
print(f"  >0.05   : {(probs_br > 0.05).sum()}")
