"""Metric helpers for Phase 5 supervised classification."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_top_k_accuracy(probabilities: np.ndarray, y_true: np.ndarray, k: int) -> float:
    if probabilities.size == 0:
        return 0.0
    top_k = np.argsort(-probabilities, axis=1)[:, :k]
    hits = [int(target) in row for target, row in zip(y_true, top_k)]
    return float(np.mean(hits))


def compute_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
) -> dict:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    roc_auc_macro_ovr = None
    present_labels = sorted(set(int(value) for value in np.asarray(y_true).tolist()))
    try:
        if probabilities.shape[1] >= 2 and len(present_labels) >= 2:
            roc_auc_macro_ovr = float(
                roc_auc_score(
                    y_true,
                    probabilities[:, present_labels],
                    average="macro",
                    multi_class="ovr",
                    labels=present_labels,
                )
            )
    except Exception:
        roc_auc_macro_ovr = None

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
        "precision_weighted": float(weighted_precision),
        "recall_weighted": float(weighted_recall),
        "f1_weighted": float(weighted_f1),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "top3_accuracy": float(compute_top_k_accuracy(probabilities, y_true, k=min(3, probabilities.shape[1]))),
        "roc_auc_macro_ovr": roc_auc_macro_ovr,
    }
    return metrics


def build_prediction_rows(
    *,
    examples: list[dict],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
) -> list[dict]:
    rows: list[dict] = []
    top_order = np.argsort(-probabilities, axis=1)
    for example, true_idx, pred_idx, probs, order in zip(
        examples, y_true, y_pred, probabilities, top_order
    ):
        top3 = [label_names[int(index)] for index in order[: min(3, len(label_names))]]
        rows.append(
            {
                "row_index": example["row_index"],
                "id": example["id"],
                "label": label_names[int(true_idx)],
                "predicted_label": label_names[int(pred_idx)],
                "predicted_confidence": float(probs[int(pred_idx)]),
                "is_correct": int(int(true_idx) == int(pred_idx)),
                "top3_labels": " | ".join(top3),
                "clustering_representation": example["clustering_representation"],
                "classifier_representation": example["classifier_representation"],
                "propagated_cluster_id": example["propagated_cluster_id"],
                "propagation_confidence": example["propagation_confidence"],
            }
        )
    return rows


def build_confusion_rows(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> list[dict]:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    rows: list[dict] = []
    for true_idx, true_label in enumerate(label_names):
        row = {"true_label": true_label}
        for pred_idx, pred_label in enumerate(label_names):
            row[pred_label] = int(matrix[true_idx, pred_idx])
        rows.append(row)
    return rows


def build_top_error_rows(
    *,
    examples: list[dict],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
    limit: int = 100,
) -> list[dict]:
    candidates: list[dict] = []
    for example, true_idx, pred_idx, probs in zip(examples, y_true, y_pred, probabilities):
        if int(true_idx) == int(pred_idx):
            continue
        candidates.append(
            {
                "row_index": example["row_index"],
                "id": example["id"],
                "label": label_names[int(true_idx)],
                "predicted_label": label_names[int(pred_idx)],
                "predicted_confidence": float(probs[int(pred_idx)]),
                "propagated_cluster_id": example["propagated_cluster_id"],
                "propagation_confidence": example["propagation_confidence"],
                "base_text_num_chars": len(example["base_text"]),
            }
        )
    candidates.sort(key=lambda item: (-item["predicted_confidence"], item["row_index"]))
    return candidates[:limit]


def compute_label_counts(labels: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(labels).items()))
