from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


def compute_metrics(
    y_true,
    probs,
    threshold=0.5
):

    preds = (probs >= threshold)

    return {
        "accuracy":
            accuracy_score(y_true, preds),

        "roc_auc":
            roc_auc_score(y_true, probs),

        "precision":
            precision_score(y_true, preds),

        "recall":
            recall_score(y_true, preds),

        "f1":
            f1_score(y_true, preds)
    }