from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "proxy_model.yaml"
ABLAT_ROOT = REPO_ROOT / "results" / "ablation"
CONFIG_ROOT = ABLAT_ROOT / "_configs"
SUMMARY_PATH = ABLAT_ROOT / "ablation_summary.csv"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def write_config(base: dict, variant: str) -> Path:
    cfg = copy.deepcopy(base)
    out_dir = ABLAT_ROOT / variant
    cfg["paths"]["output_dir"] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    path = CONFIG_ROOT / f"{variant}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def variant_config(base: dict, variant: str) -> dict:
    cfg = copy.deepcopy(base)
    cfg.setdefault("feature_flags", {})
    if variant == "full":
        cfg["feature_flags"] = {
            "terrain": True,
            "ndvi_candidates": True,
            "ndvi_groups": True,
            "soilgrids": True,
            "era5": True,
        }
    elif variant == "no_recent_ndvi":
        cfg["feature_flags"] = {
            "terrain": True,
            "ndvi_candidates": True,
            "ndvi_groups": True,
            "soilgrids": True,
            "era5": True,
        }
        cfg["paths"]["ndvi_groups"] = {
            "long_1981_2014": cfg["paths"]["ndvi_groups"]["long_1981_2014"],
        }
    elif variant == "long_ndvi_only":
        cfg["feature_flags"] = {
            "terrain": False,
            "ndvi_candidates": False,
            "ndvi_groups": True,
            "soilgrids": False,
            "era5": False,
        }
        cfg["paths"]["ndvi_candidates"] = []
        cfg["paths"]["ndvi_groups"] = {
            "long_1981_2014": cfg["paths"]["ndvi_groups"]["long_1981_2014"],
        }
        cfg["paths"]["soilgrids"] = {}
        cfg["paths"]["era5"] = {"features": {}}
    elif variant == "era5_only":
        cfg["feature_flags"] = {
            "terrain": False,
            "ndvi_candidates": False,
            "ndvi_groups": False,
            "soilgrids": False,
            "era5": True,
        }
        cfg["paths"]["ndvi_candidates"] = []
        cfg["paths"]["ndvi_groups"] = {}
        cfg["paths"]["soilgrids"] = {}
    elif variant == "terrain_only":
        cfg["feature_flags"] = {
            "terrain": True,
            "ndvi_candidates": False,
            "ndvi_groups": False,
            "soilgrids": False,
            "era5": False,
        }
        cfg["paths"]["ndvi_candidates"] = []
        cfg["paths"]["ndvi_groups"] = {}
        cfg["paths"]["soilgrids"] = {}
        cfg["paths"]["era5"] = {"features": {}}
    else:
        raise ValueError(variant)
    return cfg


def main() -> None:
    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    variants = [
        "full",
        "no_recent_ndvi",
        "long_ndvi_only",
        "era5_only",
        "terrain_only",
    ]

    summary_frames = []
    for variant in variants:
        cfg = variant_config(base, variant)
        cfg_path = write_config(cfg, variant)
        print(f"\n=== Running {variant} ===")
        run([sys.executable, str(REPO_ROOT / "src" / "classification" / "build_proxy_dataset.py"), "--config", str(cfg_path)])
        run([sys.executable, str(REPO_ROOT / "src" / "classification" / "train_baselines.py"), "--config", str(cfg_path)])
        run([sys.executable, str(REPO_ROOT / "src" / "classification" / "train_spatiotemporal.py"), "--config", str(cfg_path)])
        run([sys.executable, str(REPO_ROOT / "src" / "classification" / "evaluate.py"), "--config", str(cfg_path)])

        table_dir = Path(cfg["paths"]["output_dir"]) / "tables"
        comp = pd.read_csv(table_dir / "model_comparison.csv")
        comp.insert(0, "variant", variant)
        summary_frames.append(comp)

    ABLAT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.concat(summary_frames, ignore_index=True).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"\nWrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
