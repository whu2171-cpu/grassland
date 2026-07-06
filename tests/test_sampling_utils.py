import numpy as np

from src.classification.raster_utils import balanced_indices, quantile_classes


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
