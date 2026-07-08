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
    sample_netcdf_temporal_feature,
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


def family_enabled(config: dict, name: str, default: bool = True) -> bool:
    return bool(config.get("feature_flags", {}).get(name, default))


def add_soilgrids_features(feature_df: pd.DataFrame, config: dict, xs: np.ndarray, ys: np.ndarray, crs) -> pd.DataFrame:
    for name, path in config["paths"].get("soilgrids", {}).items():
        if not Path(path).exists():
            print(f"SKIP SoilGrids feature missing: {path}")
            continue
        values = sample_raster_at_points(path, xs, ys, crs)
        if values is not None and np.isfinite(values).any():
            feature_df[name] = values
        else:
            print(f"SKIP SoilGrids feature empty at sample points: {name}")
    return feature_df


def add_era5_features(feature_df: pd.DataFrame, config: dict, xs: np.ndarray, ys: np.ndarray) -> pd.DataFrame:
    era5_cfg = config["paths"].get("era5", {})
    default_years = years_from_config(era5_cfg, [2012, 2013, 2014])
    default_months = months_from_config(era5_cfg)
    for name, spec in era5_cfg.get("features", {}).items():
        path = spec["path"]
        if not Path(path).exists():
            print(f"SKIP ERA5 feature missing: {path}")
            continue
        years = years_from_config(spec, default_years)
        months = months_from_config(spec, default_months)
        values = sample_netcdf_temporal_feature(
            path=path,
            variable=spec["variable"],
            xs=xs,
            ys=ys,
            years=years,
            months=months,
            reducer=spec.get("reducer", "mean"),
        )
        feature_df[name] = values
    return feature_df


def years_from_config(config: dict, default: list[int] | None = None) -> list[int]:
    if "years" in config:
        return [int(year) for year in config["years"]]
    if "start_year" in config and "end_year" in config:
        return list(range(int(config["start_year"]), int(config["end_year"]) + 1))
    if default is None:
        raise ValueError("Year configuration requires years or start_year/end_year")
    return default


def months_from_config(config: dict, default: list[int] | None = None) -> list[int] | None:
    if "months" not in config:
        return default
    months = config.get("months")
    if months is None:
        return None
    return [int(month) for month in months]


def yearly_raster_paths(group_cfg: dict) -> list[tuple[int, Path]]:
    years = years_from_config(group_cfg)
    directory = Path(group_cfg["directory"])
    pattern = group_cfg["pattern"]
    paths = []
    for year in years:
        path = directory / pattern.format(year=year)
        if path.exists():
            paths.append((year, path))
        else:
            print(f"SKIP NDVI group year missing: {path}")
    return paths


def temporal_trend(values: np.ndarray, years: list[int]) -> np.ndarray:
    x = np.asarray(years, dtype="float32")
    x = x - x.mean()
    denom = np.sum(x**2)
    if denom == 0:
        return np.full(values.shape[1], np.nan, dtype="float32")
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    y = np.where(finite, values, np.nan)
    y_mean = np.nanmean(y, axis=0)
    centered = np.where(finite, y - y_mean, 0)
    slope = np.sum(centered * x[:, None], axis=0) / denom
    slope[counts < 2] = np.nan
    return slope.astype("float32")


def add_ndvi_group_features(
    feature_df: pd.DataFrame,
    config: dict,
    xs: np.ndarray,
    ys: np.ndarray,
    crs,
) -> pd.DataFrame:
    for group_name, group_cfg in config["paths"].get("ndvi_groups", {}).items():
        sampled = []
        years = []
        for year, path in yearly_raster_paths(group_cfg):
            values = sample_raster_at_points(path, xs, ys, crs)
            if values is None:
                continue
            sampled.append(values)
            years.append(year)
        if not sampled:
            print(f"SKIP NDVI group {group_name}: no available rasters")
            continue
        stack = np.vstack(sampled).astype("float32")
        if not np.isfinite(stack).any():
            print(f"SKIP NDVI group {group_name}: all sampled values are empty")
            continue
        prefix = f"ndvi_{group_name}"
        feature_df[f"{prefix}_mean"] = np.nanmean(stack, axis=0)
        feature_df[f"{prefix}_min"] = np.nanmin(stack, axis=0)
        feature_df[f"{prefix}_max"] = np.nanmax(stack, axis=0)
        feature_df[f"{prefix}_std"] = np.nanstd(stack, axis=0)
        feature_df[f"{prefix}_range"] = feature_df[f"{prefix}_max"] - feature_df[f"{prefix}_min"]
        feature_df[f"{prefix}_trend"] = temporal_trend(stack, years)
    return feature_df


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
    if family_enabled(config, "terrain"):
        dem_values = sample_raster_at_points(config["paths"]["dem"], xs, ys, label_profile["crs"])
        if dem_values is not None:
            feature_df["dem"] = dem_values
        terrain_df = sample_dem_terrain_at_points(config["paths"]["dem"], xs, ys, label_profile["crs"])
        if terrain_df is not None:
            feature_df = pd.concat([feature_df.reset_index(drop=True), terrain_df.reset_index(drop=True)], axis=1)
    ndvi_series = []
    if family_enabled(config, "ndvi_candidates"):
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
    if family_enabled(config, "ndvi_groups"):
        feature_df = add_ndvi_group_features(feature_df, config, xs, ys, label_profile["crs"])
    if family_enabled(config, "soilgrids"):
        feature_df = add_soilgrids_features(feature_df, config, xs, ys, label_profile["crs"])
    if family_enabled(config, "era5"):
        feature_df = add_era5_features(feature_df, config, xs, ys)
    df = pd.DataFrame({"task": "subtype", "row": rows, "col": cols, "x": xs, "y_coord": ys, "label": y})
    return pd.concat([df, feature_df.reset_index(drop=True)], axis=1)


def build_ndvi_proxy_samples(config: dict) -> pd.DataFrame:
    if not family_enabled(config, "ndvi_candidates"):
        return pd.DataFrame()
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
