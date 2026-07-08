from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def valid_stats(path: Path, block_size: int = 512) -> dict:
    with rasterio.open(path) as src:
        total = 0
        valid = 0
        min_value = np.inf
        max_value = -np.inf
        for row in range(0, src.height, block_size):
            for col in range(0, src.width, block_size):
                height = min(block_size, src.height - row)
                width = min(block_size, src.width - col)
                array = src.read(1, window=Window(col, row, width, height)).astype("float32")
                if src.nodata is not None and np.isfinite(src.nodata):
                    array[array == src.nodata] = np.nan
                mask = np.isfinite(array)
                total += array.size
                valid += int(mask.sum())
                if mask.any():
                    min_value = min(min_value, float(np.nanmin(array[mask])))
                    max_value = max(max_value, float(np.nanmax(array[mask])))
        return {
            "path": str(path),
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "valid": valid,
            "total": total,
            "valid_rate": valid / total if total else np.nan,
            "min": None if valid == 0 else min_value,
            "max": None if valid == 0 else max_value,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Raster files or directories to check")
    parser.add_argument("--pattern", default="*.tif")
    args = parser.parse_args()

    files: list[Path] = []
    for item in args.paths:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob(args.pattern)))
        else:
            files.append(path)

    for path in files:
        stats = valid_stats(path)
        print(
            f"{stats['path']}\tvalid_rate={stats['valid_rate']:.6f}\t"
            f"valid={stats['valid']}\tsize={stats['width']}x{stats['height']}\t"
            f"min={stats['min']}\tmax={stats['max']}\tcrs={stats['crs']}"
        )


if __name__ == "__main__":
    main()

