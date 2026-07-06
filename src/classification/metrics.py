from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def confusion_matrix_array(y_true: np.ndarray, y_pred: np.ndarray, labels: Iterable[int]) -> np.ndarray:
    labels = list(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        if truth in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[truth], label_to_idx[pred]] += 1
    return cm


def confusion_matrix_df(y_true: np.ndarray, y_pred: np.ndarray, labels: Iterable[int]) -> pd.DataFrame:
    labels = list(labels)
    cm = confusion_matrix_array(y_true, y_pred, labels)
    return pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Iterable[int]) -> dict:
    labels = list(labels)
    cm = confusion_matrix_array(y_true, y_pred, labels)
    total = cm.sum()
    correct = np.trace(cm)
    per_class = {}
    f1_scores = []

    for idx, label in enumerate(labels):
        tp = cm[idx, idx]
        row_sum = cm[idx, :].sum()
        col_sum = cm[:, idx].sum()
        producer = _safe_divide(tp, row_sum)
        user = _safe_divide(tp, col_sum)
        f1 = _safe_divide(2 * producer * user, producer + user)
        f1_scores.append(f1)
        per_class[int(label)] = {
            "support": int(row_sum),
            "producer_accuracy": producer,
            "user_accuracy": user,
            "f1": f1,
        }

    return {
        "overall_accuracy": _safe_divide(correct, total),
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "per_class": per_class,
    }


def per_class_metrics_df(metrics: dict) -> pd.DataFrame:
    rows = []
    for label, values in metrics["per_class"].items():
        row = {"class": label}
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("class")
