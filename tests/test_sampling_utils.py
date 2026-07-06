import numpy as np
import pandas as pd

from src.classification.raster_utils import balanced_indices, quantile_classes, terrain_metrics_from_window
from src.classification.split import spatial_block_ids, train_validation_indices


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


def test_spatial_block_ids_group_nearby_points():
    x = np.array([10, 20, 120, 130])
    y = np.array([5, 15, 5, 15])
    blocks = spatial_block_ids(x, y, block_size=100)

    assert blocks[0] == blocks[1]
    assert blocks[2] == blocks[3]
    assert blocks[0] != blocks[2]


def test_train_validation_indices_spatial_block_keeps_blocks_separate():
    df = pd.DataFrame(
        {
            "x": [0, 10, 100, 110, 200, 210, 300, 310],
            "y_coord": [0, 10, 0, 10, 0, 10, 0, 10],
            "label": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    config = {
        "sampling": {"seed": 42},
        "split": {"mode": "spatial_block", "validation_fraction": 0.5, "block_size": 100},
    }

    train_idx, val_idx, split_name = train_validation_indices(df, config)
    blocks = spatial_block_ids(df["x"].to_numpy(), df["y_coord"].to_numpy(), block_size=100)

    assert split_name == "spatial_block_100"
    assert set(blocks[train_idx]).isdisjoint(set(blocks[val_idx]))
    assert len(train_idx) + len(val_idx) == len(df)


def test_terrain_metrics_from_window_returns_positive_slope():
    window = np.array(
        [
            [100.0, 101.0, 102.0],
            [100.0, 101.0, 102.0],
            [100.0, 101.0, 102.0],
        ]
    )

    slope, aspect, twi_proxy = terrain_metrics_from_window(window, x_resolution_m=30.0, y_resolution_m=30.0)

    assert slope > 0
    assert 0 <= aspect < 360
    assert np.isfinite(twi_proxy)
