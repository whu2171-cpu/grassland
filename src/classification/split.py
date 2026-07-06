from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def format_block_size(block_size: float) -> str:
    return f"{block_size:g}".replace(".", "p")


def spatial_block_ids(x: np.ndarray, y: np.ndarray, block_size: float) -> np.ndarray:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    x_min = np.nanmin(x)
    y_min = np.nanmin(y)
    block_x = np.floor((x - x_min) / block_size).astype("int64")
    block_y = np.floor((y - y_min) / block_size).astype("int64")
    return np.asarray([f"{bx}_{by}" for bx, by in zip(block_x, block_y)])


def train_validation_indices(
    df: pd.DataFrame,
    config: dict,
    label_column: str = "label",
) -> tuple[np.ndarray, np.ndarray, str]:
    split_cfg = config.get("split", {})
    mode = split_cfg.get("mode", "random")
    validation_fraction = float(split_cfg.get("validation_fraction", 0.3))
    seed = int(config.get("sampling", {}).get("seed", 42))
    indices = np.arange(len(df))
    labels = df[label_column].to_numpy()

    if mode == "random":
        train_idx, val_idx = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=seed,
            stratify=labels,
        )
        return train_idx, val_idx, "random"

    if mode != "spatial_block":
        raise ValueError(f"Unsupported split mode: {mode}")

    required = {"x", "y_coord"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"spatial_block split requires columns: {sorted(missing)}")

    block_size = float(split_cfg.get("block_size", 50000))
    block_ids = spatial_block_ids(df["x"].to_numpy(), df["y_coord"].to_numpy(), block_size)
    block_df = pd.DataFrame({"block": block_ids, "label": labels})
    block_labels = (
        block_df.groupby("block")["label"]
        .agg(lambda values: pd.Series(values).mode().iloc[0])
        .rename("majority_label")
        .reset_index()
    )

    stratify = None
    label_counts = block_labels["majority_label"].value_counts()
    if len(block_labels) >= 4 and (label_counts >= 2).all():
        stratify = block_labels["majority_label"].to_numpy()

    if stratify is not None:
        train_blocks, val_blocks = train_test_split(
            block_labels["block"].to_numpy(),
            test_size=validation_fraction,
            random_state=seed,
            stratify=stratify,
        )
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
        split = splitter.split(indices, labels, groups=block_ids)
        train_idx, val_idx = next(split)
        return train_idx, val_idx, f"spatial_block_{format_block_size(block_size)}"

    val_block_set = set(val_blocks.tolist())
    is_val = np.asarray([block in val_block_set for block in block_ids])
    train_idx = indices[~is_val]
    val_idx = indices[is_val]
    return train_idx, val_idx, f"spatial_block_{format_block_size(block_size)}"
