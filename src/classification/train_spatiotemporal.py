from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classification.config import load_config
from src.classification.metrics import classification_metrics, confusion_matrix_df, per_class_metrics_df
from src.classification.split import train_validation_indices
from src.classification.train_baselines import feature_columns


def train_mlp_for_task(df: pd.DataFrame, task: str, config: dict, table_dir: Path) -> dict | None:
    task_df = df[df["task"] == task].copy()
    if task_df.empty:
        return None
    features = feature_columns(task_df)
    if not features:
        return None

    x_df = task_df[features].replace([np.inf, -np.inf], np.nan)
    x = x_df.fillna(x_df.median()).to_numpy()
    y = task_df["label"].to_numpy()
    labels = sorted(np.unique(y).tolist())
    train_idx, val_idx, split_name = train_validation_indices(task_df.reset_index(drop=True), config)
    train_x, val_x = x[train_idx], x[val_idx]
    train_y, val_y = y[train_idx], y[val_idx]

    mlp_cfg = config["models"]["mlp"]
    model = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=tuple(int(v) for v in mlp_cfg["hidden_layer_sizes"]),
            max_iter=int(mlp_cfg["max_iter"]),
            random_state=int(mlp_cfg["random_state"]),
            early_stopping=True,
        ),
    )
    model.fit(train_x, train_y)
    pred = model.predict(val_x)
    metrics = classification_metrics(val_y, pred, labels)
    model_name = "mlp_prototype"
    confusion_matrix_df(val_y, pred, labels).to_csv(
        table_dir / f"confusion_matrix_{task}_{model_name}.csv",
        encoding="utf-8-sig",
    )
    per_class_metrics_df(metrics).to_csv(
        table_dir / f"class_metrics_{task}_{model_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "task": task,
        "model": model_name,
        "label_source": "proxy",
        "n_train": len(train_y),
        "n_val": len(val_y),
        "split": split_name,
        "features": ";".join(features),
        "overall_accuracy": metrics["overall_accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/proxy_model.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    table_dir = Path(config["paths"]["output_dir"]) / "tables"

    frames = []
    for name in ["proxy_samples_subtype.csv", "proxy_samples_fvc_ndvi.csv"]:
        path = table_dir / name
        if path.exists():
            try:
                frames.append(pd.read_csv(path))
            except EmptyDataError:
                continue
    if not frames:
        raise FileNotFoundError("No proxy sample tables found. Run build_proxy_dataset.py first.")
    df = pd.concat(frames, ignore_index=True)

    rows = []
    for task in sorted(df["task"].dropna().unique().tolist()):
        row = train_mlp_for_task(df, task, config, table_dir)
        if row is not None:
            rows.append(row)
    pd.DataFrame(rows).to_csv(table_dir / "model_comparison_neural.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {table_dir / 'model_comparison_neural.csv'}")


if __name__ == "__main__":
    main()
