"""Tests for trace validation, heuristic collection, and dataset persistence."""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from data_collector import (
    CollectorConfig,
    collect,
    load_trace,
    parse_nodes,
    save_dataset,
)
from train_offline_rl import load_dataset


def test_parse_nodes_rejects_malformed_or_nonfinite_values() -> None:
    np.testing.assert_array_equal(
        parse_nodes("4:16,8:32"), np.asarray([[4, 16], [8, 32]], dtype=np.float32)
    )
    for value in ("4", "0:16", "nan:16", "4:-1"):
        with pytest.raises(Exception):
            parse_nodes(value)


def test_load_trace_sanitizes_and_drops_oversized_jobs(tmp_path) -> None:
    trace = tmp_path / "trace.csv"
    pd.DataFrame(
        {
            "submitted": [2.0, 0.0, 1.0, 3.0, 4.0],
            "cpu": [2.0, 1.0, 99.0, -1.0, 1.0],
            "memory": [4.0, 2.0, 1.0, 2.0, np.nan],
            "runtime": [3.0, 1.0, 1.0, 1.0, 1.0],
            "sla": [6.0, np.nan, 2.0, 2.0, 2.0],
        }
    ).to_csv(trace, index=False)
    args = Namespace(
        submission_column="submitted",
        cpu_column="cpu",
        memory_column="memory",
        duration_column="runtime",
        sla_column="sla",
        sla_multiplier=2.0,
        nodes=np.asarray([[4.0, 16.0]], dtype=np.float32),
    )
    workload = load_trace(trace, args)
    assert workload.shape == (2, 5)
    assert workload[:, 0].tolist() == [0.0, 2.0]
    assert workload[0, 4] == 2.0  # Missing SLA defaults to duration * multiplier.


@pytest.mark.parametrize("extension", [".npz", ".h5"])
def test_collection_and_dataset_round_trip(tmp_path, extension: str) -> None:
    workload = np.asarray(
        [
            [0.0, 1.0, 2.0, 2.0, 4.0],
            [0.0, 2.0, 4.0, 1.0, 4.0],
            [1.0, 1.0, 1.0, 1.0, 4.0],
            [2.0, 3.0, 4.0, 2.0, 4.0],
        ],
        dtype=np.float64,
    )
    config = CollectorConfig(
        queue_capacity=2,
        node_capacities=np.asarray([[4.0, 16.0], [8.0, 32.0]], dtype=np.float32),
        heuristic="sjf",
        jobs_per_episode=2,
        seed=11,
    )
    arrays = collect(workload, config)
    assert arrays["states"].shape[1] == 2 * 5 + 2 * 6 + 4
    assert arrays["action_masks"].shape[1] == 2 * 2 + 1
    assert arrays["dones"].sum() == 2
    rows = np.arange(arrays["actions"].shape[0])
    assert np.all(arrays["action_masks"][rows, arrays["actions"]])

    output = tmp_path / f"offline{extension}"
    metadata = {"schema_version": 1, "heuristic": "sjf"}
    save_dataset(output, arrays, metadata)
    assert output.is_file()
    assert not output.with_name(output.name + ".tmp").exists()
    loaded, loaded_metadata = load_dataset(output)
    assert loaded_metadata == metadata
    for name, expected in arrays.items():
        np.testing.assert_array_equal(loaded[name], expected)
