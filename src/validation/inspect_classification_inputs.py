"""Inspect candidate labels and subtype classification rasters.

This script is intentionally read-only. It summarizes:
- Hulunbuir_Labeled_Points.csv label-like fields.
- Subtype_Classification*.tif class distributions.
- Subtype_Confidence*.tif confidence distributions.

Run from repository root:
    python src/validation/inspect_classification_inputs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


OUTPUT_DIR = Path(r"D:\草地遥感\result\汇报\260513\output")
CSV_PATH = OUTPUT_DIR / "Hulunbuir_Labeled_Points.csv"
RASTERS = [
    "Subtype_Classification.tif",
    "Subtype_Classification_Full.tif",
    "Subtype_Confidence.tif",
    "Subtype_Confidence_Full.tif",
]


def print_csv_summary(path: Path) -> None:
    print("=" * 80)
    print(f"CSV: {path}")
    df = pd.read_csv(path)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    for col in ["landcover", "Y_cluster", "dist_level"]:
        if col in df.columns:
            print(f"\nValue counts: {col}")
            print(df[col].value_counts(dropna=False).sort_index())

    numeric_cols = [
        "ndvi_mean",
        "ndvi_cv",
        "ndvi_trend",
        "precip",
        "temp",
        "elevation",
        "slope",
        "dist_human",
        "night",
    ]
    existing = [c for c in numeric_cols if c in df.columns]
    if existing:
        print("\nNumeric summary:")
        print(df[existing].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T)


def print_class_raster_summary(path: Path) -> None:
    print("=" * 80)
    print(f"Raster: {path.name}")
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata
        mask = np.ones(arr.shape, dtype=bool)
        if nodata is not None and np.isfinite(nodata):
            mask &= arr != nodata
        vals, counts = np.unique(arr[mask], return_counts=True)
        total = counts.sum()
        print(
            f"shape={src.height}x{src.width}, count={src.count}, "
            f"crs={src.crs}, dtype={src.dtypes[0]}, nodata={nodata}"
        )
        print("class,count,percent")
        for val, count in zip(vals, counts):
            print(f"{int(val)},{int(count)},{count / total * 100:.4f}")


def print_confidence_raster_summary(path: Path) -> None:
    print("=" * 80)
    print(f"Raster: {path.name}")
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        arr = arr[np.isfinite(arr)]
        q = np.nanpercentile(arr, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        print(
            f"shape={src.height}x{src.width}, count={src.count}, "
            f"crs={src.crs}, dtype={src.dtypes[0]}, n={arr.size:,}"
        )
        print(f"min={np.nanmin(arr):.6f}, mean={np.nanmean(arr):.6f}, max={np.nanmax(arr):.6f}")
        print("quantiles 1/5/10/25/50/75/90/95/99:")
        print(", ".join(f"{x:.6f}" for x in q))


def main() -> None:
    print_csv_summary(CSV_PATH)
    for name in RASTERS:
        path = OUTPUT_DIR / name
        if "Confidence" in name:
            print_confidence_raster_summary(path)
        else:
            print_class_raster_summary(path)


if __name__ == "__main__":
    main()

