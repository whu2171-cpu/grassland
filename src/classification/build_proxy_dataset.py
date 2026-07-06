from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classification.config import load_config
from src.classification.raster_utils import (
    balanced_indices,
    pixel_centers,
    quantile_classes,
    read_single_band,
    sample_dem_terrain_at_points,
    sample_raster_at_points,
    sample_feature_values,
    valid_mask,
)


def same_grid(profile_a: dict, profile_b: dict, shape_a: tuple[int, int], shape_b: tuple[int, int]) -> bool:
    return (
        shape_a == shape_b
        and profile_a["transform"] == profile_b["transform"]
        and str(profile_a["crs"]) == str(profile_b["crs"])
    )


def load_feature_if_same_grid(
    name: str,
    path: str,
    base_profile: dict,
    base_shape: tuple[int, int],
) -> tuple[str, np.ndarray] | None:
    array, profile = read_single_band(path)
    if not same_grid(base_profile, profile, base_shape, array.shape):
        print(f"SKIP feature {name}: grid mismatch for {path}")
        return None
    return name, array.astype("float32")


def build_subtype_samples(config: dict) -> pd.DataFrame:
    label, label_profile = read_single_band(config["paths"]["subtype_label"])
    confidence, confidence_profile = read_single_band(config["paths"]["subtype_confidence"])
    if not same_grid(label_profile, confidence_profile, label.shape, confidence.shape):
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

    xs, ys = pixel_centers(rows, cols, label_profile["transform"])
    features: dict[str, np.ndarray] = {"confidence": confidence.astype("float32")}
    feature_df = sample_feature_values(features, rows, cols)
    dem_values = sample_raster_at_points(config["paths"]["dem"], xs, ys, label_profile["crs"])
    if dem_values is not None:
        feature_df["dem"] = dem_values
    terrain_df = sample_dem_terrain_at_points(config["paths"]["dem"], xs, ys, label_profile["crs"])
    if terrain_df is not None:
        feature_df = pd.concat([feature_df.reset_index(drop=True), terrain_df.reset_index(drop=True)], axis=1)
    ndvi_series = []
    for ndvi_path in config["paths"].get("ndvi_candidates", []):
        path = Path(ndvi_path)
        if not path.exists():
            print(f"SKIP subtype NDVI feature missing: {ndvi_path}")
            continue
        values = sample_raster_at_points(path, xs, ys, label_profile["crs"])
        if values is None:
            continue
        name = path.stem.lower().replace("-", "_")
        feature_df[f"{name}_point"] = values
        ndvi_series.append(values)
    if ndvi_series:
        ndvi_stack = np.vstack(ndvi_series).astype("float32")
        feature_df["ndvi_time_mean"] = np.nanmean(ndvi_stack, axis=0)
        feature_df["ndvi_time_min"] = np.nanmin(ndvi_stack, axis=0)
        feature_df["ndvi_time_max"] = np.nanmax(ndvi_stack, axis=0)
        feature_df["ndvi_time_std"] = np.nanstd(ndvi_stack, axis=0)
        feature_df["ndvi_time_range"] = feature_df["ndvi_time_max"] - feature_df["ndvi_time_min"]
    df = pd.DataFrame({"task": "subtype", "row": rows, "col": cols, "x": xs, "y_coord": ys, "label": y})
    return pd.concat([df, feature_df.reset_index(drop=True)], axis=1)


def build_ndvi_proxy_samples(config: dict) -> pd.DataFrame:
    candidates = config["paths"].get("ndvi_candidates", [])
    arrays = []
    base_profile = None
    base_shape = None
    for path in candidates:
        if not Path(path).exists():
            print(f"SKIP NDVI candidate missing: {path}")
            continue
        array, profile = read_single_band(path)
        if base_profile is None:
            base_profile = profile
            base_shape = array.shape
        elif not same_grid(base_profile, profile, base_shape, array.shape):
            print(f"SKIP NDVI candidate grid mismatch: {path}")
            continue
        arrays.append(array.astype("float32"))

    if not arrays or base_profile is None:
        return pd.DataFrame()

    stack = np.stack(arrays, axis=0)
    finite = np.isfinite(stack)
    count = finite.sum(axis=0)
    summed = np.where(finite, stack, 0).sum(axis=0)
    mean_ndvi = np.full(stack.shape[1:], np.nan, dtype="float32")
    mean_ndvi[count > 0] = summed[count > 0] / count[count > 0]
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
