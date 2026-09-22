#!/usr/bin/env python3
"""Collect an offline scheduling dataset from CSV/Parquet cluster traces."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from hpc_scheduler import ClusterSchedulingEnv, JOB_FEATURE_COUNT, NODE_FEATURE_COUNT


@dataclass(frozen=True)
class CollectorConfig:
    """Immutable parameters controlling trajectory generation."""

    queue_capacity: int
    node_capacities: np.ndarray
    heuristic: Literal["sjf", "drf"]
    jobs_per_episode: int
    seed: int


def parse_nodes(value: str) -> np.ndarray:
    """Parse ``cpu:memory,cpu:memory`` into an ``[N, 2]`` array."""
    try:
        nodes = [[float(part) for part in item.split(":")] for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("nodes must use cpu:memory pairs") from exc
    result = np.asarray(nodes, dtype=np.float32)
    if (
        result.ndim != 2
        or result.shape[1] != 2
        or not np.all(np.isfinite(result))
        or np.any(result <= 0)
    ):
        raise argparse.ArgumentTypeError("nodes must contain positive cpu:memory pairs")
    return result


def load_trace(path: Path, args: argparse.Namespace) -> np.ndarray:
    """Read, sanitize, and canonicalize a workload trace."""
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".csv.gz"} or path.name.endswith(".csv.gz"):
        frame = pd.read_csv(path)
    else:
        raise ValueError("trace must be CSV, CSV.GZ, or Parquet")

    required = [args.submission_column, args.cpu_column, args.memory_column, args.duration_column]
    if args.sla_column:
        required.append(args.sla_column)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"trace is missing columns: {', '.join(missing)}")
    selected = frame[required[:4]].apply(pd.to_numeric, errors="coerce")
    selected.columns = ["submission", "cpu", "memory", "duration"]
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna()
    selected = selected[
        (selected["submission"] >= 0)
        & (selected["cpu"] > 0)
        & (selected["memory"] > 0)
        & (selected["duration"] > 0)
    ]
    if args.sla_column:
        sla = pd.to_numeric(frame.loc[selected.index, args.sla_column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        selected["sla"] = sla.fillna(selected["duration"] * args.sla_multiplier)
    else:
        selected["sla"] = selected["duration"] * args.sla_multiplier
    selected["sla"] = selected["sla"].clip(lower=0)
    selected = selected.sort_values("submission", kind="stable")

    capacities = args.nodes
    can_fit = np.asarray(
        [np.any((capacities[:, 0] >= row.cpu) & (capacities[:, 1] >= row.memory))
         for row in selected.itertuples()],
        dtype=bool,
    )
    dropped = int((~can_fit).sum())
    if dropped:
        print(f"warning: dropping {dropped} jobs that cannot fit any node", file=sys.stderr)
    result = selected.loc[can_fit, ["submission", "cpu", "memory", "duration", "sla"]]
    if result.empty:
        raise ValueError("no valid jobs remain after trace validation")
    return result.to_numpy(dtype=np.float64, copy=True)


def choose_action(
    observation: np.ndarray,
    mask: np.ndarray,
    node_count: int,
    queue_capacity: int,
    heuristic: Literal["sjf", "drf"],
) -> int:
    """Select a valid SJF or dominant-resource best-fit action."""
    valid = np.flatnonzero(mask[:-1])
    if valid.size == 0:
        return int(mask.size - 1)
    node_offset = node_count * NODE_FEATURE_COUNT
    scored: list[tuple[float, float, int]] = []
    for action in valid.tolist():
        job_idx, node_idx = divmod(action, node_count)
        if job_idx >= queue_capacity:
            continue
        job_offset = node_offset + job_idx * JOB_FEATURE_COUNT
        cpu_req, mem_req = observation[job_offset : job_offset + 2]
        duration = float(observation[job_offset + 2])
        node_cpu_util = float(observation[node_idx * 5])
        node_mem_util = float(observation[node_idx * 5 + 1])
        post_utilization = max(node_cpu_util + cpu_req, node_mem_util + mem_req)
        if heuristic == "sjf":
            scored.append((duration, post_utilization, action))
        else:
            dominant_request = max(float(cpu_req), float(mem_req))
            # DRF-like ordering, with tight best-fit placement as the tie breaker.
            scored.append((dominant_request, -post_utilization, action))
    return min(scored)[2]


def collect(workload: np.ndarray, config: CollectorConfig) -> dict[str, np.ndarray]:
    """Run heuristic episodes and return transition arrays."""
    fields: dict[str, list[np.ndarray | float | int | bool]] = {
        "states": [], "actions": [], "rewards": [], "next_states": [],
        "dones": [], "action_masks": [], "next_action_masks": [], "episode_ids": [],
    }
    episode_returns: list[float] = []
    episode_id = 0
    for start in range(0, len(workload), config.jobs_per_episode):
        episode = workload[start : start + config.jobs_per_episode].copy()
        episode[:, 0] -= episode[0, 0]
        env = ClusterSchedulingEnv(
            config.node_capacities,
            config.queue_capacity,
            episode,
        )
        observation, info = env.reset(seed=config.seed + episode_id)
        total_reward = 0.0
        done = False
        while not done:
            mask = np.asarray(info["action_mask"], dtype=bool)
            action = choose_action(
                observation,
                mask,
                config.node_capacities.shape[0],
                config.queue_capacity,
                config.heuristic,
            )
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            fields["states"].append(observation.copy())
            fields["actions"].append(action)
            fields["rewards"].append(reward)
            fields["next_states"].append(next_observation.copy())
            fields["dones"].append(done)
            fields["action_masks"].append(mask.copy())
            fields["next_action_masks"].append(
                np.asarray(next_info["action_mask"], dtype=bool).copy()
            )
            fields["episode_ids"].append(episode_id)
            observation, info = next_observation, next_info
            total_reward += reward
        episode_returns.append(total_reward)
        episode_id += 1

    arrays = {
        "states": np.asarray(fields["states"], dtype=np.float32),
        "actions": np.asarray(fields["actions"], dtype=np.int64),
        "rewards": np.asarray(fields["rewards"], dtype=np.float32),
        "next_states": np.asarray(fields["next_states"], dtype=np.float32),
        "dones": np.asarray(fields["dones"], dtype=np.bool_),
        "action_masks": np.asarray(fields["action_masks"], dtype=np.bool_),
        "next_action_masks": np.asarray(fields["next_action_masks"], dtype=np.bool_),
        "episode_ids": np.asarray(fields["episode_ids"], dtype=np.int64),
        "episode_returns": np.asarray(episode_returns, dtype=np.float32),
    }
    return arrays


def save_dataset(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    """Persist a compressed NPZ or HDF5 dataset via an atomic same-directory replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(metadata, sort_keys=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        if path.suffix.lower() == ".npz":
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays, metadata=np.asarray(metadata_json))
        elif path.suffix.lower() in {".h5", ".hdf5"}:
            import h5py

            with h5py.File(temporary, "w") as output:
                for name, values in arrays.items():
                    output.create_dataset(name, data=values, compression="gzip", shuffle=True)
                output.attrs["metadata"] = metadata_json
        else:
            raise ValueError("output extension must be .npz, .h5, or .hdf5")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/offline_dataset.npz"))
    parser.add_argument("--nodes", type=parse_nodes, default=parse_nodes("32:128,48:192,64:256"))
    parser.add_argument("--queue-capacity", type=int, default=32)
    parser.add_argument("--jobs-per-episode", type=int, default=10_000)
    parser.add_argument("--heuristic", choices=("sjf", "drf"), default="sjf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--submission-column", default="submission_time")
    parser.add_argument("--cpu-column", default="cpu_req")
    parser.add_argument("--memory-column", default="mem_req")
    parser.add_argument("--duration-column", default="duration")
    parser.add_argument("--sla-column", default=None)
    parser.add_argument("--sla-multiplier", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.queue_capacity <= 0 or args.jobs_per_episode <= 0:
        raise ValueError("queue-capacity and jobs-per-episode must be positive")
    if not np.isfinite(args.sla_multiplier) or args.sla_multiplier < 0.0:
        raise ValueError("sla-multiplier must be finite and nonnegative")
    workload = load_trace(args.trace, args)
    config = CollectorConfig(
        queue_capacity=args.queue_capacity,
        node_capacities=args.nodes,
        heuristic=args.heuristic,
        jobs_per_episode=args.jobs_per_episode,
        seed=args.seed,
    )
    arrays = collect(workload, config)
    metadata = {
        "schema_version": 1,
        "heuristic": args.heuristic,
        "queue_capacity": args.queue_capacity,
        "node_capacities": args.nodes.tolist(),
        "state_dim": int(arrays["states"].shape[1]),
        "action_dim": int(arrays["action_masks"].shape[1]),
        "transitions": int(arrays["states"].shape[0]),
        "episodes": int(arrays["episode_returns"].shape[0]),
        "seed": args.seed,
    }
    save_dataset(args.output, arrays, metadata)
    print(
        f"wrote {metadata['transitions']} transitions from {metadata['episodes']} episodes "
        f"to {args.output} (mean return={arrays['episode_returns'].mean():.4f})"
    )


if __name__ == "__main__":
    main()
