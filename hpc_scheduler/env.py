"""Farama Gymnasium wrapper around the pybind11 state engine."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from hpc_scheduler import _cluster_env
except ImportError as exc:  # pragma: no cover - exercised only before native build
    raise ImportError(
        "The native extension is not built. Run `pip install -e .` or build with CMake."
    ) from exc

NODE_FEATURE_COUNT = int(_cluster_env.NODE_FEATURE_COUNT)
JOB_FEATURE_COUNT = int(_cluster_env.JOB_FEATURE_COUNT)
GLOBAL_FEATURE_COUNT = int(_cluster_env.GLOBAL_FEATURE_COUNT)


class ClusterSchedulingEnv(gym.Env[np.ndarray, int]):
    """Event-driven scheduling environment with a flattened discrete action space.

    Actions ``0 .. K*N-1`` map to ``queue_slot * node_count + node``. The final
    action is WAIT, which advances simulation time to the next arrival/completion.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        node_capacities: np.ndarray,
        queue_capacity: int,
        workload: np.ndarray | None = None,
        *,
        alpha: float = 1.0,
        beta: float = 0.25,
        gamma: float = 2.0,
        invalid_action_penalty: float = 2.0,
        time_scale: float = 3600.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        capacities = np.asarray(node_capacities, dtype=np.float32, order="C")
        if (
            capacities.ndim != 2
            or capacities.shape[1] != 2
            or not np.all(np.isfinite(capacities))
            or np.any(capacities <= 0)
        ):
            raise ValueError("node_capacities must have shape [N, 2] with positive finite values")
        if render_mode not in {None, "ansi"}:
            raise ValueError("render_mode must be None or 'ansi'")
        reward = _cluster_env.RewardConfig()
        reward.alpha = alpha
        reward.beta = beta
        reward.gamma = gamma
        reward.invalid_action_penalty = invalid_action_penalty
        reward.time_scale = time_scale
        self._native = _cluster_env.ClusterEnv(capacities, queue_capacity, reward)
        self._workload = self._validate_workload(workload) if workload is not None else None
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(self._native.action_count)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._native.state_size,),
            dtype=np.float32,
        )

    @staticmethod
    def _validate_workload(workload: np.ndarray) -> np.ndarray:
        result = np.asarray(workload, dtype=np.float64, order="C")
        if result.ndim != 2 or result.shape[1] != 5 or not np.all(np.isfinite(result)):
            raise ValueError(
                "workload must be [jobs, 5]: submission_time,cpu_req,mem_req,duration,sla_wait"
            )
        return result

    def set_workload(self, workload: np.ndarray) -> None:
        """Replace the workload used by subsequent calls to :meth:`reset`."""
        self._workload = self._validate_workload(workload)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the engine with the configured or ``options['workload']`` trace."""
        super().reset(seed=seed)
        workload = options.get("workload") if options else self._workload
        if workload is None:
            raise ValueError("set a workload before reset or pass options={'workload': array}")
        self._workload = self._validate_workload(workload)
        observation, info = self._native.reset(self._workload, seed or 0)
        info["action_mask"] = self.action_mask()
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply a flattened placement action or the final WAIT action."""
        if not self.action_space.contains(action):
            raise ValueError(f"action {action} is outside {self.action_space}")
        if action == self.action_space.n - 1:
            job_idx, node_idx = -1, -1
        else:
            job_idx, node_idx = divmod(int(action), self._native.node_count)
        observation, reward, terminated, truncated, info = self._native.step(
            job_idx, node_idx
        )
        info["action_mask"] = self.action_mask()
        return observation, float(reward), terminated, truncated, info

    def action_mask(self) -> np.ndarray:
        """Return a boolean mask over valid flattened actions."""
        return self._native.action_mask()

    @property
    def state_view(self) -> np.ndarray:
        """Expose native state without copying; consume it before the next step."""
        view = self._native.state_view()
        view.flags.writeable = False
        return view

    def render(self) -> str:
        """Return a compact ANSI-compatible cluster status line."""
        return str(self._native.render())
