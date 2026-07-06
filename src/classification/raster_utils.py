from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


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
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("values must contain finite numbers")
    quantiles = np.linspace(0, 1, n_classes + 1)[1:-1]
    bins = np.quantile(finite, quantiles)
    return np.digitize(values, bins, right=False).astype(np.int16)


def pixel_centers(rows: np.ndarray, cols: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    return np.asarray(xs), np.asarray(ys)


def sample_feature_values(feature_arrays: dict[str, np.ndarray], rows: np.ndarray, cols: np.ndarray) -> pd.DataFrame:
    data = {}
    for name, array in feature_arrays.items():
        data[name] = array[rows, cols]
    return pd.DataFrame(data)


def sample_raster_at_points(path: str | Path, xs: np.ndarray, ys: np.ndarray, expected_crs) -> np.ndarray | None:
    with rasterio.open(path) as src:
        if expected_crs is not None and src.crs is not None and str(src.crs) != str(expected_crs):
            print(f"SKIP coordinate sampling due to CRS mismatch: {path}")
            return None
        samples = src.sample(zip(xs.tolist(), ys.tolist()))
        values = np.asarray([sample[0] for sample in samples], dtype="float32")
        nodata = src.nodata
    if nodata is not None and np.isfinite(nodata):
        values[values == nodata] = np.nan
    return values


def terrain_metrics_from_window(
    window: np.ndarray,
    x_resolution_m: float,
    y_resolution_m: float,
) -> tuple[float, float, float]:
    if window.shape != (3, 3) or not np.isfinite(window).all():
        return np.nan, np.nan, np.nan
    dzdx = (float(window[1, 2]) - float(window[1, 0])) / (2 * x_resolution_m)
    dzdy = (float(window[2, 1]) - float(window[0, 1])) / (2 * y_resolution_m)
    gradient = np.hypot(dzdx, dzdy)
    slope_deg = float(np.degrees(np.arctan(gradient)))
    aspect_deg = float((np.degrees(np.arctan2(dzdx, -dzdy)) + 360) % 360)
    twi_proxy = float(np.log(1.0 / (np.tan(np.radians(slope_deg)) + 1e-6)))
    return slope_deg, aspect_deg, twi_proxy


def sample_dem_terrain_at_points(
    path: str | Path,
    xs: np.ndarray,
    ys: np.ndarray,
    expected_crs,
) -> pd.DataFrame | None:
    slope = np.full(len(xs), np.nan, dtype="float32")
    aspect = np.full(len(xs), np.nan, dtype="float32")
    twi_proxy = np.full(len(xs), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        if expected_crs is not None and src.crs is not None and str(src.crs) != str(expected_crs):
            print(f"SKIP terrain sampling due to CRS mismatch: {path}")
            return None
        x_degree_size = abs(src.transform.a)
        y_degree_size = abs(src.transform.e)
        nodata = src.nodata
        fill_value = nodata if nodata is not None else np.nan
        for idx, (x, y) in enumerate(zip(xs.tolist(), ys.tolist())):
            row, col = src.index(x, y)
            if row <= 0 or col <= 0 or row >= src.height - 1 or col >= src.width - 1:
                continue
            data = src.read(1, window=Window(col - 1, row - 1, 3, 3), boundless=True, fill_value=fill_value)
            data = data.astype("float32")
            if nodata is not None and np.isfinite(nodata):
                data[data == nodata] = np.nan
            latitude = np.radians(y)
            x_resolution_m = max(x_degree_size * 111_320.0 * np.cos(latitude), 1.0)
            y_resolution_m = max(y_degree_size * 110_540.0, 1.0)
            slope[idx], aspect[idx], twi_proxy[idx] = terrain_metrics_from_window(
                data,
                x_resolution_m,
                y_resolution_m,
            )
    return pd.DataFrame({"slope": slope, "aspect": aspect, "twi_proxy": twi_proxy})
