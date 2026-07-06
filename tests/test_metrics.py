import numpy as np

from src.classification.metrics import classification_metrics, confusion_matrix_df


def test_classification_metrics_handles_missing_predictions():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2])
    labels = [0, 1, 2]

    metrics = classification_metrics(y_true, y_pred, labels)

    assert round(metrics["overall_accuracy"], 6) == round(4 / 6, 6)
    assert round(metrics["macro_f1"], 6) == round((0.5 + 0.8 + 0.6666666667) / 3, 6)
    assert metrics["per_class"][0]["producer_accuracy"] == 0.5
    assert metrics["per_class"][1]["user_accuracy"] == 2 / 3
    assert metrics["per_class"][2]["producer_accuracy"] == 0.5


def test_confusion_matrix_df_shape_and_labels():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2])
    cm = confusion_matrix_df(y_true, y_pred, [0, 1, 2])

    assert list(cm.index) == ["true_0", "true_1", "true_2"]
    assert list(cm.columns) == ["pred_0", "pred_1", "pred_2"]
    assert cm.loc["true_0", "pred_0"] == 1
    assert cm.loc["true_0", "pred_1"] == 1
