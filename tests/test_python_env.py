"""Integration tests for the Gymnasium/pybind11 boundary."""

from __future__ import annotations

import numpy as np
import pytest

hpc_scheduler = pytest.importorskip("hpc_scheduler")


def test_gym_contract_and_action_mask() -> None:
    workload = np.asarray(
        [[0.0, 1.0, 2.0, 3.0, 5.0], [1.0, 2.0, 4.0, 2.0, 5.0]],
        dtype=np.float64,
    )
    environment = hpc_scheduler.ClusterSchedulingEnv(
        np.asarray([[4.0, 16.0]], dtype=np.float32), 2, workload
    )
    state, info = environment.reset(seed=7)
    assert environment.observation_space.contains(state)
    assert info["action_mask"].shape == (3,)
    assert info["action_mask"][0]
    read_only_view = environment.state_view
    assert not read_only_view.flags.writeable
    state, reward, terminated, truncated, info = environment.step(0)
    assert np.isfinite(reward)
    assert environment.observation_space.contains(state)
    assert not truncated
    assert isinstance(terminated, bool)
    assert "completed_jobs" in info


def test_invalid_action_is_penalized_and_wait_progresses_time() -> None:
    workload = np.asarray([[0.0, 3.0, 2.0, 2.0, 5.0]], dtype=np.float64)
    environment = hpc_scheduler.ClusterSchedulingEnv(
        np.asarray([[2.0, 8.0], [4.0, 8.0]], dtype=np.float32),
        1,
        workload,
        invalid_action_penalty=7.0,
    )
    _, info = environment.reset()
    assert info["action_mask"].tolist() == [False, True, False]
    initial_time = environment._native.current_time
    _, reward, terminated, _, info = environment.step(0)
    assert reward <= -7.0
    assert not terminated
    assert info["action_valid"] is False
    assert environment._native.current_time == initial_time

    _, _, terminated, _, _ = environment.step(1)
    assert terminated
    assert environment._native.completed_jobs == 1


def test_configuration_and_terminal_guards() -> None:
    with pytest.raises(ValueError):
        hpc_scheduler.ClusterSchedulingEnv(np.asarray([[0.0, 8.0]], dtype=np.float32), 1)
    environment = hpc_scheduler.ClusterSchedulingEnv(
        np.asarray([[2.0, 8.0]], dtype=np.float32),
        1,
        np.asarray([[0.0, 1.0, 1.0, 1.0, 2.0]], dtype=np.float64),
    )
    environment.reset()
    environment.step(0)
    with pytest.raises(RuntimeError, match="terminal"):
        environment.step(0)
