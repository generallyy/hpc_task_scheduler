"""Unit tests for policy selection and temporal holdout behavior."""

from __future__ import annotations

import numpy as np
import pytest

from evaluate_policy import choose_heuristic_action, select_temporal_holdout


def test_temporal_holdout_selects_latest_jobs_and_applies_cap() -> None:
    workload = np.column_stack(
        (
            np.arange(10, dtype=np.float64),
            np.ones((10, 4), dtype=np.float64),
        )
    )
    selected = select_temporal_holdout(workload, 0.5, 3)
    assert selected[:, 0].tolist() == [7.0, 8.0, 9.0]
    selected[0, 0] = -1.0
    assert workload[7, 0] == 7.0
    with pytest.raises(ValueError):
        select_temporal_holdout(workload, 0.0, None)


def test_heuristics_only_return_valid_placements() -> None:
    # One node (5 features), two queue jobs (6 features each), four globals.
    observation = np.asarray(
        [
            0.25, 0.25, 1.0, 1.0, 0.1,
            0.25, 0.25, 0.1, 0.4, 0.2, 1.0,
            0.50, 0.25, 0.5, 0.1, 0.3, 1.0,
            0.0, 1.0, 0.0, 0.25,
        ],
        dtype=np.float32,
    )
    mask = np.asarray([True, True, True])
    for heuristic in ("fifo", "sjf", "drf", "best_fit"):
        action = choose_heuristic_action(observation, mask, 1, 2, heuristic)
        assert action in (0, 1)
        assert mask[action]


def test_heuristic_waits_only_when_no_placement_exists() -> None:
    observation = np.zeros(5 + 6 + 4, dtype=np.float32)
    mask = np.asarray([False, True])
    assert choose_heuristic_action(observation, mask, 1, 1, "sjf") == 1
    with pytest.raises(RuntimeError, match="no valid"):
        choose_heuristic_action(observation, np.asarray([False, False]), 1, 1, "sjf")
