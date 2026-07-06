from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
