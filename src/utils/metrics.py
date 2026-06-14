from sklearn.metrics import roc_auc_score, f1_score

def evaluate(F, hiv_active, mask):
    """
    mask: boolean mask of nodes to evaluate
    """
    probs = F[:, 1] / (F.sum(dim=1) + 1e-8)

    y_true = hiv_active[mask].cpu().numpy()
    y_pred = probs[mask].detach().cpu().numpy()

    auc = roc_auc_score(y_true, y_pred)

    preds_bin = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_true, preds_bin)

    return {"AUC": auc, "F1": f1}