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
        values = [sample[0] for sample in samples]
    return np.asarray(values, dtype="float32")
