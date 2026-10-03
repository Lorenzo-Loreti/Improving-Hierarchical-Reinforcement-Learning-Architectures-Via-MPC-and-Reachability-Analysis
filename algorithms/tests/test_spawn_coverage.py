"""Tests for algorithms/spawn_coverage.py, the spawn/* metrics the training
loops log every update."""

import numpy as np
from scipy.stats import qmc

from spawn_coverage import SpawnCoverage

LOW, HIGH = np.array([0.0, -1.0]), np.array([2.0, 1.0])  # the 4 m corridors' spawn box


def _to_box(unit):
    """Unit-square points as observations in the spawn box, at rest."""
    unit = np.asarray(unit, dtype=np.float64)
    return np.column_stack([LOW + unit * (HIGH - LOW), np.zeros((len(unit), 2))])


def test_no_metrics_until_the_window_is_full():
    coverage = SpawnCoverage(LOW, HIGH, window=8)
    coverage.add(_to_box(np.full((7, 2), 0.5)))
    assert coverage.metrics() == {}
    coverage.add(_to_box(np.full((1, 2), 0.5)))
    assert set(coverage.metrics()) == {"spawn/discrepancy", "spawn/empty_cells"}


def test_empty_cells_counts_the_5x5_cells_no_start_fell_in():
    centres = (np.stack(np.meshgrid(np.arange(5), np.arange(5)), axis=-1).reshape(-1, 2) + 0.5) / 5
    spread = SpawnCoverage(LOW, HIGH, window=25)
    spread.add(_to_box(centres))
    assert spread.metrics()["spawn/empty_cells"] == 0
    clustered = SpawnCoverage(LOW, HIGH, window=25)
    clustered.add(_to_box(np.full((25, 2), 0.1)))
    assert clustered.metrics()["spawn/empty_cells"] == 24


def test_discrepancy_is_scipys_on_the_unit_square_and_lower_when_spread():
    rng = np.random.default_rng(0)
    unit = rng.random((32, 2))
    coverage = SpawnCoverage(LOW, HIGH)
    coverage.add(_to_box(unit))
    np.testing.assert_allclose(coverage.metrics()["spawn/discrepancy"],
                               qmc.discrepancy(unit, method="CD"))
    sobol = SpawnCoverage(LOW, HIGH)
    sobol.add(_to_box(qmc.Sobol(2, rng=np.random.default_rng(0)).random(32)))
    assert sobol.metrics()["spawn/discrepancy"] < coverage.metrics()["spawn/discrepancy"]


def test_window_slides_over_the_latest_starts_and_takes_empty_batches():
    coverage = SpawnCoverage(LOW, HIGH, window=4)
    coverage.add(_to_box(np.full((10, 2), 0.05)))      # old, clustered in one corner
    coverage.add(np.zeros((0, 4)))                     # a step on which no episode ended
    coverage.add(_to_box([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]))
    assert coverage.metrics()["spawn/empty_cells"] == 21  # only the four corner cells


def test_starts_on_the_upper_edge_count_in_the_last_cell():
    coverage = SpawnCoverage(LOW, HIGH, window=1)
    coverage.add(np.array([[2.0, 1.0, 0.0, 0.0]], dtype=np.float32))
    assert coverage.metrics()["spawn/empty_cells"] == 24
