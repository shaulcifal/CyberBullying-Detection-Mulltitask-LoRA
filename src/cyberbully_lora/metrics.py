import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

IGNORE_INDEX = -100


def classification_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
    }


def rationale_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Token-level rationale metrics, following the HateXplain-style evaluation:
    plain token F1/precision/recall, plus IOU-F1 (intersection-over-union >= 0.5
    between predicted and gold rationale spans counts as a match).
    """
    flat_preds, flat_labels = [], []
    iou_scores = []

    for pred_row, label_row in zip(preds, labels):
        p_seq, l_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l == IGNORE_INDEX:
                continue
            p_seq.append(int(p))
            l_seq.append(int(l))
        flat_preds.extend(p_seq)
        flat_labels.extend(l_seq)

        pred_set = {i for i, v in enumerate(p_seq) if v == 1}
        gold_set = {i for i, v in enumerate(l_seq) if v == 1}
        if pred_set or gold_set:
            union = len(pred_set | gold_set)
            inter = len(pred_set & gold_set)
            iou_scores.append(inter / union if union else 0.0)

    token_metrics = {
        "token_accuracy": accuracy_score(flat_labels, flat_preds),
        "token_f1": f1_score(flat_labels, flat_preds, average="binary", zero_division=0),
        "token_precision": precision_score(flat_labels, flat_preds, average="binary", zero_division=0),
        "token_recall": recall_score(flat_labels, flat_preds, average="binary", zero_division=0),
    }
    iou_f1_at_50 = float(np.mean([1.0 if s >= 0.5 else 0.0 for s in iou_scores])) if iou_scores else 0.0
    token_metrics["mean_iou"] = float(np.mean(iou_scores)) if iou_scores else 0.0
    token_metrics["iou_f1_at_0.5"] = iou_f1_at_50
    return token_metrics
