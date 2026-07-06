# Proxy Model Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimum runnable proxy classification pipeline using subtype pseudo-labels and optional FVC/NDVI proxy labels.

**Architecture:** A config-driven Python pipeline will build raster samples, train traditional baselines, train a small neural prototype, and export evaluation tables. The first version uses simple raster feature extraction and tabular models so the engineering chain runs before full Sentinel-2 U-Net + Transformer training.

**Tech Stack:** Python, numpy, pandas, rasterio, scikit-learn, optional torch, YAML config.

---

## File Structure

- Create `configs/proxy_model.yaml`: local paths, sampling settings, split settings, model settings.
- Create `src/classification/__init__.py`: package marker.
- Create `src/classification/config.py`: lightweight YAML config loader.
- Create `src/classification/metrics.py`: confusion matrix and OA/PA/UA/F1 metrics.
- Create `src/classification/raster_utils.py`: raster read, sampling, coordinate, and alignment helpers.
- Create `src/classification/build_proxy_dataset.py`: build subtype and NDVI proxy sample tables.
- Create `src/classification/train_baselines.py`: train Random Forest and sklearn gradient boosting.
- Create `src/classification/train_spatiotemporal.py`: train MLP prototype with sklearn fallback if torch is unavailable.
- Create `src/classification/evaluate.py`: evaluate saved predictions or sample tables.
- Create `tests/test_metrics.py`: metric correctness tests.
- Create `tests/test_sampling_utils.py`: sampling utility tests.

## Task 1: Metrics Utilities

**Files:**
- Create: `src/classification/__init__.py`
- Create: `src/classification/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write tests**

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_metrics.py -v`

Expected: FAIL because `src.classification.metrics` does not exist.

- [ ] **Step 3: Implement metrics**

Create `src/classification/__init__.py` as an empty file.

Create `src/classification/metrics.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
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
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_metrics.py -v`

Expected: PASS.

## Task 2: Config And Sampling Utilities

**Files:**
- Create: `src/classification/config.py`
- Create: `src/classification/raster_utils.py`
- Create: `tests/test_sampling_utils.py`

- [ ] **Step 1: Write tests**

```python
import numpy as np

from src.classification.raster_utils import balanced_indices, quantile_classes


def test_balanced_indices_limits_each_class():
    labels = np.array([0, 0, 0, 1, 1, 2])
    idx = balanced_indices(labels, max_per_class=2, seed=7)
    sampled = labels[idx]

    assert len(sampled) == 5
    assert (sampled == 0).sum() == 2
    assert (sampled == 1).sum() == 2
    assert (sampled == 2).sum() == 1


def test_quantile_classes_builds_three_classes():
    values = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    classes = quantile_classes(values, n_classes=3)

    assert set(classes.tolist()) == {0, 1, 2}
    assert len(classes) == len(values)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_sampling_utils.py -v`

Expected: FAIL because `raster_utils` does not exist.

- [ ] **Step 3: Implement config and raster utilities**

Create `src/classification/config.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data
```

Create `src/classification/raster_utils.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def read_single_band(path: str | Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        array = src.read(1)
        profile = src.profile.copy()
        profile["transform"] = src.transform
        profile["crs"] = src.crs
        profile["nodata"] = src.nodata
    return array, profile


def valid_mask(array: np.ndarray, nodata: float | int | None) -> np.ndarray:
    mask = np.isfinite(array)
    if nodata is not None and np.isfinite(nodata):
        mask &= array != nodata
    return mask


def balanced_indices(labels: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(np.unique(labels).tolist()):
        class_idx = np.flatnonzero(labels == label)
        if len(class_idx) > max_per_class:
            class_idx = rng.choice(class_idx, size=max_per_class, replace=False)
        selected.append(class_idx)
    if not selected:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate(selected).astype(np.int64))


def quantile_classes(values: np.ndarray, n_classes: int) -> np.ndarray:
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    quantiles = np.linspace(0, 1, n_classes + 1)[1:-1]
    bins = np.quantile(values[np.isfinite(values)], quantiles)
    return np.digitize(values, bins, right=False).astype(np.int16)


def pixel_centers(rows: np.ndarray, cols: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    return np.asarray(xs), np.asarray(ys)


def sample_feature_values(feature_arrays: dict[str, np.ndarray], rows: np.ndarray, cols: np.ndarray) -> pd.DataFrame:
    data = {}
    for name, array in feature_arrays.items():
        data[name] = array[rows, cols]
    return pd.DataFrame(data)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_sampling_utils.py -v`

Expected: PASS.

## Task 3: Dataset Builder And Config

**Files:**
- Create: `configs/proxy_model.yaml`
- Create: `src/classification/build_proxy_dataset.py`

- [ ] **Step 1: Add config**

Create `configs/proxy_model.yaml` with current local paths and conservative sample sizes:

```yaml
paths:
  output_dir: "D:/草地遥感/中期/grassland-remote-sensing/results"
  subtype_label: "D:/草地遥感/result/汇报/260513/output/Subtype_Classification_Full.tif"
  subtype_confidence: "D:/草地遥感/result/汇报/260513/output/Subtype_Confidence_Full.tif"
  dem: "D:/草地遥感/result/汇报/260513/dem/Mongolian_Plateau_DEM.tif"
  ndvi_candidates:
    - "D:/草地遥感/result/tiff/2020_ndvi.tif"
    - "D:/草地遥感/result/tiff/2021_ndvi.tif"
    - "D:/草地遥感/result/tiff/2022_ndvi.tif"

sampling:
  seed: 42
  max_per_class: 3000
  confidence_threshold: 0.95
  fallback_confidence_threshold: 0.90
  fvc_ndvi_classes: 5

split:
  validation_fraction: 0.30
  mode: "random"

models:
  random_forest:
    n_estimators: 200
    max_depth: 20
    random_state: 42
  gradient_boosting:
    max_iter: 200
    learning_rate: 0.08
    random_state: 42
  mlp:
    hidden_layer_sizes: [128, 64]
    max_iter: 200
    random_state: 42
```

- [ ] **Step 2: Implement dataset builder**

Create `src/classification/build_proxy_dataset.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.classification.config import load_config
from src.classification.raster_utils import (
    balanced_indices,
    pixel_centers,
    quantile_classes,
    read_single_band,
    sample_feature_values,
    valid_mask,
)


def _same_grid(profile_a: dict, profile_b: dict, shape_a: tuple[int, int], shape_b: tuple[int, int]) -> bool:
    return (
        shape_a == shape_b
        and profile_a["transform"] == profile_b["transform"]
        and str(profile_a["crs"]) == str(profile_b["crs"])
    )


def _load_feature_if_same_grid(name: str, path: str, base_profile: dict, base_shape: tuple[int, int]) -> tuple[str, np.ndarray] | None:
    array, profile = read_single_band(path)
    if not _same_grid(base_profile, profile, base_shape, array.shape):
        print(f"SKIP feature {name}: grid mismatch for {path}")
        return None
    return name, array.astype("float32")


def build_subtype_samples(config: dict) -> pd.DataFrame:
    label, label_profile = read_single_band(config["paths"]["subtype_label"])
    confidence, confidence_profile = read_single_band(config["paths"]["subtype_confidence"])
    if not _same_grid(label_profile, confidence_profile, label.shape, confidence.shape):
        raise ValueError("Subtype label and confidence rasters are not aligned")

    threshold = float(config["sampling"]["confidence_threshold"])
    mask = valid_mask(label, label_profile.get("nodata"))
    mask &= valid_mask(confidence, confidence_profile.get("nodata"))
    mask &= confidence >= threshold
    mask &= np.isin(label, [0, 1, 2, 3, 4])

    rows, cols = np.where(mask)
    y = label[rows, cols].astype("int16")
    selected = balanced_indices(y, int(config["sampling"]["max_per_class"]), int(config["sampling"]["seed"]))
    rows, cols, y = rows[selected], cols[selected], y[selected]

    features = {}
    dem_item = _load_feature_if_same_grid("dem", config["paths"]["dem"], label_profile, label.shape)
    if dem_item is not None:
        features[dem_item[0]] = dem_item[1]
    features["confidence"] = confidence.astype("float32")

    feature_df = sample_feature_values(features, rows, cols)
    xs, ys = pixel_centers(rows, cols, label_profile["transform"])
    df = pd.DataFrame({"task": "subtype", "row": rows, "col": cols, "x": xs, "y_coord": ys, "label": y})
    return pd.concat([df, feature_df.reset_index(drop=True)], axis=1)


def build_ndvi_proxy_samples(config: dict) -> pd.DataFrame:
    candidates = config["paths"].get("ndvi_candidates", [])
    arrays = []
    base_profile = None
    for path in candidates:
        if not Path(path).exists():
            print(f"SKIP NDVI candidate missing: {path}")
            continue
        array, profile = read_single_band(path)
        if base_profile is None:
            base_profile = profile
            base_shape = array.shape
        elif not _same_grid(base_profile, profile, base_shape, array.shape):
            print(f"SKIP NDVI candidate grid mismatch: {path}")
            continue
        arrays.append(array.astype("float32"))

    if not arrays:
        return pd.DataFrame()

    stack = np.stack(arrays, axis=0)
    mean_ndvi = np.nanmean(stack, axis=0)
    mask = np.isfinite(mean_ndvi)
    rows, cols = np.where(mask)
    values = mean_ndvi[rows, cols]
    labels = quantile_classes(values, int(config["sampling"]["fvc_ndvi_classes"]))
    selected = balanced_indices(labels, int(config["sampling"]["max_per_class"]), int(config["sampling"]["seed"]))
    rows, cols, labels, values = rows[selected], cols[selected], labels[selected], values[selected]
    xs, ys = pixel_centers(rows, cols, base_profile["transform"])
    return pd.DataFrame(
        {
            "task": "fvc_ndvi",
            "row": rows,
            "col": cols,
            "x": xs,
            "y_coord": ys,
            "label": labels,
            "mean_ndvi": values,
        }
    )


def write_summary(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        summary = pd.DataFrame(columns=["task", "label", "count"])
    else:
        summary = df.groupby(["task", "label"]).size().reset_index(name="count")
    summary.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/proxy_model.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["paths"]["output_dir"])
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    subtype_df = build_subtype_samples(config)
    subtype_path = table_dir / "proxy_samples_subtype.csv"
    subtype_df.to_csv(subtype_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {subtype_path} rows={len(subtype_df)}")

    ndvi_df = build_ndvi_proxy_samples(config)
    ndvi_path = table_dir / "proxy_samples_fvc_ndvi.csv"
    ndvi_df.to_csv(ndvi_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {ndvi_path} rows={len(ndvi_df)}")

    combined = pd.concat([subtype_df, ndvi_df], ignore_index=True)
    write_summary(combined, table_dir / "proxy_sample_summary.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke run dataset builder**

Run: `python src/classification/build_proxy_dataset.py --config configs/proxy_model.yaml`

Expected: Writes subtype sample CSV. NDVI proxy may be written or empty if candidate files are missing/misaligned.

## Task 4: Traditional Baselines

**Files:**
- Create: `src/classification/train_baselines.py`

- [ ] **Step 1: Implement baselines**

```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.classification.config import load_config
from src.classification.metrics import classification_metrics, confusion_matrix_df, per_class_metrics_df


NON_FEATURE_COLUMNS = {"task", "row", "col", "x", "y_coord", "label"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[col]):
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

    x = task_df[features].replace([np.inf, -np.inf], np.nan).fillna(task_df[features].median()).to_numpy()
    y = task_df["label"].to_numpy()
    labels = sorted(np.unique(y).tolist())
    train_x, val_x, train_y, val_y = train_test_split(
        x,
        y,
        test_size=float(config["split"]["validation_fraction"]),
        random_state=int(config["sampling"]["seed"]),
        stratify=y,
    )

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
```

- [ ] **Step 2: Smoke run baselines**

Run: `python src/classification/train_baselines.py --config configs/proxy_model.yaml`

Expected: Writes model comparison and confusion matrix CSV files.

## Task 5: Neural Prototype

**Files:**
- Create: `src/classification/train_spatiotemporal.py`

- [ ] **Step 1: Implement MLP prototype**

```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.classification.config import load_config
from src.classification.metrics import classification_metrics, confusion_matrix_df, per_class_metrics_df
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
    train_x, val_x, train_y, val_y = train_test_split(
        x,
        y,
        test_size=float(config["split"]["validation_fraction"]),
        random_state=int(config["sampling"]["seed"]),
        stratify=y,
    )

    mlp_cfg = config["models"]["mlp"]
    model = make_pipeline(
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
            frames.append(pd.read_csv(path))
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
```

- [ ] **Step 2: Smoke run neural prototype**

Run: `python src/classification/train_spatiotemporal.py --config configs/proxy_model.yaml`

Expected: Writes MLP comparison and metric CSV files.

## Task 6: Combined Evaluation

**Files:**
- Create: `src/classification/evaluate.py`

- [ ] **Step 1: Implement combined comparison writer**

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.classification.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/proxy_model.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    table_dir = Path(config["paths"]["output_dir"]) / "tables"

    frames = []
    for name in ["model_comparison_baselines.csv", "model_comparison_neural.csv"]:
        path = table_dir / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No model comparison files found.")
    comparison = pd.concat(frames, ignore_index=True).sort_values(["task", "macro_f1"], ascending=[True, False])
    out = table_dir / "model_comparison.csv"
    comparison.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run combined evaluation**

Run: `python src/classification/evaluate.py --config configs/proxy_model.yaml`

Expected: Writes `results/tables/model_comparison.csv`.

## Task 7: Documentation And Verification

**Files:**
- Modify: `CHANGELOG.md`
- Create: `docs/研究日志/2026-07-06_模型链路最小实现记录.md`

- [ ] **Step 1: Run full tests**

Run: `python -m pytest tests -v`

Expected: PASS.

- [ ] **Step 2: Run full smoke pipeline**

Run:

```powershell
python src/classification/build_proxy_dataset.py --config configs/proxy_model.yaml
python src/classification/train_baselines.py --config configs/proxy_model.yaml
python src/classification/train_spatiotemporal.py --config configs/proxy_model.yaml
python src/classification/evaluate.py --config configs/proxy_model.yaml
```

Expected:

- `results/tables/proxy_sample_summary.csv`
- `results/tables/model_comparison_baselines.csv`
- `results/tables/model_comparison_neural.csv`
- `results/tables/model_comparison.csv`

- [ ] **Step 3: Write implementation record**

Create `docs/研究日志/2026-07-06_模型链路最小实现记录.md` with:

```markdown
# 模型链路最小实现记录

日期：2026-07-06

## 完成内容

- 建立 proxy model 配置；
- 建立 subtype 高置信伪标签样本构建脚本；
- 建立 FVC/NDVI proxy label 样本构建接口；
- 实现 Random Forest、HistGradientBoosting 和 MLP prototype；
- 输出混淆矩阵、OA、PA、UA、F1、macro-F1 和模型对比表。

## 边界说明

当前结果属于 proxy/internal validation，不代表最终草地亚型独立验证精度。

## 下一步

- 增加环境因子栅格；
- 改为空间 block split；
- 替换/补充真实草地亚型 reference samples；
- 实现正式 U-Net + Transformer 模型。
```

- [ ] **Step 4: Commit implementation**

Run:

```bash
git add configs src tests results/tables docs/研究日志 CHANGELOG.md
git commit -m "Add proxy model chain prototype"
```

Expected: Commit succeeds.
