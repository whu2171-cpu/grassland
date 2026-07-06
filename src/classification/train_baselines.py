from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classification.config import load_config
from src.classification.metrics import classification_metrics, confusion_matrix_df, per_class_metrics_df
from src.classification.split import train_validation_indices


NON_FEATURE_COLUMNS = {"task", "row", "col", "x", "y_coord", "label"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[col]) and not df[col].isna().all():
            cols.append(col)
    return cols


def train_one_task(df: pd.DataFrame, task: str, config: dict, table_dir: Path) -> list[dict]:
    task_df = df[df["task"] == task].copy()
    if task_df.empty:
        print(f"SKIP task {task}: no samples")
        return []
    features = feature_columns(task_df)
    if not features:
        print(f"SKIP task {task}: no numeric features")
        return []

    x_df = task_df[features].replace([np.inf, -np.inf], np.nan)
    x = x_df.fillna(x_df.median()).to_numpy()
    y = task_df["label"].to_numpy()
    labels = sorted(np.unique(y).tolist())
    train_idx, val_idx, split_name = train_validation_indices(task_df.reset_index(drop=True), config)
    train_x, val_x = x[train_idx], x[val_idx]
    train_y, val_y = y[train_idx], y[val_idx]

    rf_cfg = config["models"]["random_forest"]
    gb_cfg = config["models"]["gradient_boosting"]
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=int(rf_cfg["n_estimators"]),
            max_depth=int(rf_cfg["max_depth"]),
            random_state=int(rf_cfg["random_state"]),
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=int(gb_cfg["max_iter"]),
            learning_rate=float(gb_cfg["learning_rate"]),
            random_state=int(gb_cfg["random_state"]),
        ),
    }

    rows = []
    for model_name, model in models.items():
        model.fit(train_x, train_y)
        pred = model.predict(val_x)
        metrics = classification_metrics(val_y, pred, labels)
        confusion_matrix_df(val_y, pred, labels).to_csv(
            table_dir / f"confusion_matrix_{task}_{model_name}.csv",
            encoding="utf-8-sig",
        )
        per_class_metrics_df(metrics).to_csv(
            table_dir / f"class_metrics_{task}_{model_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        rows.append(
            {
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
        )

        if hasattr(model, "feature_importances_"):
            pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values(
                "importance", ascending=False
            ).to_csv(table_dir / f"feature_importance_{task}_{model_name}.csv", index=False, encoding="utf-8-sig")
    return rows


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
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No proxy sample tables found. Run build_proxy_dataset.py first.")

    df = pd.concat(frames, ignore_index=True)
    rows = []
    for task in sorted(df["task"].dropna().unique().tolist()):
        rows.extend(train_one_task(df, task, config, table_dir))
    pd.DataFrame(rows).to_csv(table_dir / "model_comparison_baselines.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {table_dir / 'model_comparison_baselines.csv'}")


if __name__ == "__main__":
    main()
