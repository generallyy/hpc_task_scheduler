#!/usr/bin/env python3
"""Replay a held-out trace against an ONNX policy and deterministic baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import onnxruntime as ort

from hpc_scheduler import ClusterSchedulingEnv, JOB_FEATURE_COUNT, NODE_FEATURE_COUNT
from data_collector import load_trace, parse_nodes

HeuristicName = Literal["fifo", "sjf", "drf", "best_fit"]
PolicySelector = Callable[[np.ndarray, np.ndarray], int]


@dataclass(frozen=True)
class EvaluationConfig:
    """Cluster and episode parameters shared by every replayed policy."""

    node_capacities: np.ndarray
    queue_capacity: int
    jobs_per_episode: int
    seed: int = 42
    alpha: float = 1.0
    beta: float = 0.25
    gamma: float = 2.0
    invalid_action_penalty: float = 2.0
    time_scale: float = 3600.0
    max_decisions_per_job: int = 20


@dataclass(frozen=True)
class EpisodeResult:
    """Raw measurements from one deterministic replay episode."""

    policy: str
    episode_id: int
    jobs: int
    completed_jobs: int
    decisions: int
    invalid_actions: int
    total_return: float
    makespan: float
    sla_violations: int
    sla_jobs: int
    wait_times: tuple[float, ...]
    slowdowns: tuple[float, ...]
    decision_latencies_us: tuple[float, ...]
    cpu_resource_time: float
    memory_resource_time: float
    total_cpu_capacity: float
    total_memory_capacity: float

    def record(self) -> dict[str, float | int | str]:
        """Return a compact per-episode record suitable for CSV or JSON."""
        return {
            "policy": self.policy,
            "episode_id": self.episode_id,
            "jobs": self.jobs,
            "completed_jobs": self.completed_jobs,
            "decisions": self.decisions,
            "invalid_actions": self.invalid_actions,
            "total_return": self.total_return,
            "makespan": self.makespan,
            "throughput": self.completed_jobs / self.makespan if self.makespan > 0 else 0.0,
            "mean_wait": float(np.mean(self.wait_times)) if self.wait_times else 0.0,
            "p95_wait": percentile(self.wait_times, 95.0),
            "mean_slowdown": float(np.mean(self.slowdowns)) if self.slowdowns else 1.0,
            "p95_slowdown": percentile(self.slowdowns, 95.0),
            "sla_violations": self.sla_violations,
            "sla_violation_rate": (
                self.sla_violations / self.sla_jobs if self.sla_jobs else 0.0
            ),
            "cpu_utilization": safe_utilization(
                self.cpu_resource_time, self.total_cpu_capacity, self.makespan
            ),
            "memory_utilization": safe_utilization(
                self.memory_resource_time, self.total_memory_capacity, self.makespan
            ),
            "mean_decision_latency_us": (
                float(np.mean(self.decision_latencies_us))
                if self.decision_latencies_us
                else 0.0
            ),
            "p95_decision_latency_us": percentile(self.decision_latencies_us, 95.0),
        }


class OnnxPolicy:
    """Validated ONNX Runtime policy with the production state/mask contract."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = {item.name: item for item in self.session.get_inputs()}
        outputs = {item.name: item for item in self.session.get_outputs()}
        if set(inputs) != {"state", "action_mask"} or "action_logits" not in outputs:
            raise ValueError(
                "model must expose state/action_mask inputs and action_logits output"
            )
        if inputs["state"].type != "tensor(float)":
            raise ValueError("model state input must be float32")
        if inputs["action_mask"].type != "tensor(uint8)":
            raise ValueError("model action_mask input must be uint8")
        if outputs["action_logits"].type != "tensor(float)":
            raise ValueError("model action_logits output must be float32")
        self.state_dim = self._fixed_width(inputs["state"].shape, "state")
        self.action_dim = self._fixed_width(inputs["action_mask"].shape, "action_mask")
        output_dim = self._fixed_width(outputs["action_logits"].shape, "action_logits")
        if output_dim != self.action_dim:
            raise ValueError("model action mask and logit widths differ")
        # Warm graph initialization so reported decision latency excludes session startup.
        self.session.run(
            ["action_logits"],
            {
                "state": np.zeros((1, self.state_dim), dtype=np.float32),
                "action_mask": np.ones((1, self.action_dim), dtype=np.uint8),
            },
        )

    @staticmethod
    def _fixed_width(shape: Sequence[object], name: str) -> int:
        if len(shape) != 2 or not isinstance(shape[1], int) or shape[1] <= 0:
            raise ValueError(f"model {name} must be rank 2 with a fixed feature width")
        return shape[1]

    def validate_environment(self, environment: ClusterSchedulingEnv) -> None:
        """Fail before replay when model and environment schemas do not match."""
        if environment.observation_space.shape != (self.state_dim,):
            raise ValueError(
                f"model state width {self.state_dim} does not match environment "
                f"width {environment.observation_space.shape[0]}"
            )
        if environment.action_space.n != self.action_dim:
            raise ValueError(
                f"model action width {self.action_dim} does not match environment "
                f"width {environment.action_space.n}"
            )

    def __call__(self, observation: np.ndarray, mask: np.ndarray) -> int:
        logits = self.session.run(
            ["action_logits"],
            {
                "state": np.asarray(observation, dtype=np.float32)[None, :],
                "action_mask": np.asarray(mask, dtype=np.uint8)[None, :],
            },
        )[0]
        if logits.shape != (1, self.action_dim) or not np.all(np.isfinite(logits)):
            raise RuntimeError("ONNX policy returned malformed or nonfinite logits")
        return int(np.argmax(logits[0]))


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a percentile while defining the empty-set value as zero."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile)) if values else 0.0


def safe_utilization(resource_time: float, capacity: float, makespan: float) -> float:
    denominator = capacity * makespan
    return resource_time / denominator if denominator > 0.0 else 0.0


def choose_heuristic_action(
    observation: np.ndarray,
    mask: np.ndarray,
    node_count: int,
    queue_capacity: int,
    heuristic: HeuristicName,
) -> int:
    """Choose FIFO, SJF, dominant-resource-first, or two-resource best-fit."""
    valid_actions = np.flatnonzero(mask[:-1])
    if valid_actions.size == 0:
        if mask[-1]:
            return int(mask.size - 1)
        raise RuntimeError("nonterminal state has no valid placement or WAIT action")

    node_offset = node_count * NODE_FEATURE_COUNT
    scored: list[tuple[tuple[float, ...], int]] = []
    for action in valid_actions.tolist():
        job_idx, node_idx = divmod(action, node_count)
        if job_idx >= queue_capacity:
            continue
        job_offset = node_offset + job_idx * JOB_FEATURE_COUNT
        cpu_request = float(observation[job_offset])
        memory_request = float(observation[job_offset + 1])
        duration = float(observation[job_offset + 2])
        wait = float(observation[job_offset + 3])
        node_feature_offset = node_idx * NODE_FEATURE_COUNT
        cpu_utilization = float(observation[node_feature_offset])
        memory_utilization = float(observation[node_feature_offset + 1])
        cpu_capacity = float(observation[node_feature_offset + 2])
        memory_capacity = float(observation[node_feature_offset + 3])
        cpu_residual = max(0.0, cpu_capacity * (1.0 - cpu_utilization) - cpu_request)
        memory_residual = max(
            0.0, memory_capacity * (1.0 - memory_utilization) - memory_request
        )
        residual_waste = cpu_residual + memory_residual
        imbalance = abs(cpu_residual - memory_residual)
        placement_score = (residual_waste, imbalance, float(node_idx))

        if heuristic == "fifo":
            priority = (-wait, float(job_idx), *placement_score)
        elif heuristic == "sjf":
            priority = (duration, -wait, *placement_score, float(job_idx))
        elif heuristic == "drf":
            priority = (
                max(cpu_request, memory_request),
                -wait,
                *placement_score,
                float(job_idx),
            )
        elif heuristic == "best_fit":
            priority = (*placement_score, -max(cpu_request, memory_request), float(job_idx))
        else:  # Defensive even though callers and argparse constrain this value.
            raise ValueError(f"unsupported heuristic: {heuristic}")
        scored.append((priority, action))
    if not scored:
        raise RuntimeError("action mask contained no decodable placement")
    return min(scored, key=lambda item: item[0])[1]


def heuristic_selector(
    name: HeuristicName, node_count: int, queue_capacity: int
) -> PolicySelector:
    return lambda observation, mask: choose_heuristic_action(
        observation, mask, node_count, queue_capacity, name
    )


def run_episode(
    policy_name: str,
    selector: PolicySelector,
    workload: np.ndarray,
    config: EvaluationConfig,
    episode_id: int,
) -> EpisodeResult:
    """Replay one policy on one workload episode and collect direct outcomes."""
    episode = np.asarray(workload, dtype=np.float64).copy()
    if episode.ndim != 2 or episode.shape[1] != 5 or episode.shape[0] == 0:
        raise ValueError("episode workload must be a nonempty [jobs, 5] array")
    episode[:, 0] -= episode[0, 0]
    environment = ClusterSchedulingEnv(
        config.node_capacities,
        config.queue_capacity,
        episode,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
        invalid_action_penalty=config.invalid_action_penalty,
        time_scale=config.time_scale,
    )
    observation, info = environment.reset(seed=config.seed + episode_id)
    wait_times: list[float] = []
    slowdowns: list[float] = []
    latencies: list[float] = []
    total_return = 0.0
    invalid_actions = 0
    decisions = 0
    done = False
    max_decisions = max(100, len(episode) * config.max_decisions_per_job)
    final_info = info

    while not done:
        if decisions >= max_decisions:
            raise RuntimeError(
                f"policy {policy_name} exceeded {max_decisions} decisions; possible WAIT loop"
            )
        mask = np.asarray(info["action_mask"], dtype=bool)
        decision_start = time.perf_counter_ns()
        action = int(selector(observation, mask))
        latencies.append((time.perf_counter_ns() - decision_start) / 1_000.0)
        decisions += 1
        if action < 0 or action >= mask.size or not mask[action]:
            invalid_actions += 1
            valid = np.flatnonzero(mask)
            if valid.size == 0:
                raise RuntimeError("environment exposed no valid recovery action")
            action = int(valid[0])

        if action != environment.action_space.n - 1:
            job_idx = action // config.node_capacities.shape[0]
            job_offset = (
                config.node_capacities.shape[0] * NODE_FEATURE_COUNT
                + job_idx * JOB_FEATURE_COUNT
            )
            wait = float(observation[job_offset + 3]) * config.time_scale
            duration = float(observation[job_offset + 2]) * config.time_scale
            wait_times.append(wait)
            slowdowns.append((wait + duration) / duration if duration > 0.0 else 1.0)

        observation, reward, terminated, truncated, final_info = environment.step(action)
        total_return += reward
        done = terminated or truncated
        info = final_info

    makespan = float(final_info["current_time"])
    completed_jobs = int(final_info["completed_jobs"])
    if completed_jobs != len(episode) or len(wait_times) != len(episode):
        raise RuntimeError(
            f"policy {policy_name} completed {completed_jobs}/{len(episode)} jobs "
            f"and recorded {len(wait_times)} scheduling decisions"
        )
    cpu_resource_time = float(np.sum(episode[:, 1] * episode[:, 3]))
    memory_resource_time = float(np.sum(episode[:, 2] * episode[:, 3]))
    return EpisodeResult(
        policy=policy_name,
        episode_id=episode_id,
        jobs=len(episode),
        completed_jobs=completed_jobs,
        decisions=decisions,
        invalid_actions=invalid_actions,
        total_return=total_return,
        makespan=makespan,
        sla_violations=int(final_info["sla_violations"]),
        sla_jobs=int(np.count_nonzero(episode[:, 4] > 0.0)),
        wait_times=tuple(wait_times),
        slowdowns=tuple(slowdowns),
        decision_latencies_us=tuple(latencies),
        cpu_resource_time=cpu_resource_time,
        memory_resource_time=memory_resource_time,
        total_cpu_capacity=float(np.sum(config.node_capacities[:, 0])),
        total_memory_capacity=float(np.sum(config.node_capacities[:, 1])),
    )


def evaluate_workload(
    workload: np.ndarray,
    model_path: Path,
    config: EvaluationConfig,
    heuristics: Sequence[HeuristicName] = ("fifo", "sjf", "drf", "best_fit"),
) -> list[EpisodeResult]:
    """Replay every policy over identical chronological episode chunks."""
    if workload.ndim != 2 or workload.shape[1] != 5 or len(workload) == 0:
        raise ValueError("workload must be a nonempty [jobs, 5] array")
    if config.jobs_per_episode <= 0 or config.queue_capacity <= 0:
        raise ValueError("jobs_per_episode and queue_capacity must be positive")
    if config.max_decisions_per_job <= 0:
        raise ValueError("max_decisions_per_job must be positive")
    model = OnnxPolicy(model_path)
    probe = ClusterSchedulingEnv(
        config.node_capacities,
        config.queue_capacity,
        workload[:1],
        time_scale=config.time_scale,
    )
    model.validate_environment(probe)
    selectors: list[tuple[str, PolicySelector]] = [
        (name, heuristic_selector(name, config.node_capacities.shape[0], config.queue_capacity))
        for name in heuristics
    ]
    selectors.append(("onnx", model))
    results: list[EpisodeResult] = []
    for episode_id, start in enumerate(range(0, len(workload), config.jobs_per_episode)):
        episode = workload[start : start + config.jobs_per_episode]
        for policy_name, selector in selectors:
            results.append(
                run_episode(policy_name, selector, episode, config, episode_id)
            )
    return results


def aggregate_results(results: Sequence[EpisodeResult]) -> dict[str, dict[str, float | int]]:
    """Aggregate episode measurements without averaging percentiles or ratios."""
    summaries: dict[str, dict[str, float | int]] = {}
    for policy in sorted({result.policy for result in results}):
        selected = [result for result in results if result.policy == policy]
        waits = tuple(value for result in selected for value in result.wait_times)
        slowdowns = tuple(value for result in selected for value in result.slowdowns)
        latencies = tuple(value for result in selected for value in result.decision_latencies_us)
        jobs = sum(result.jobs for result in selected)
        completed = sum(result.completed_jobs for result in selected)
        total_makespan = sum(result.makespan for result in selected)
        violations = sum(result.sla_violations for result in selected)
        sla_jobs = sum(result.sla_jobs for result in selected)
        cpu_time = sum(result.cpu_resource_time for result in selected)
        memory_time = sum(result.memory_resource_time for result in selected)
        cpu_capacity_time = sum(
            result.total_cpu_capacity * result.makespan for result in selected
        )
        memory_capacity_time = sum(
            result.total_memory_capacity * result.makespan for result in selected
        )
        summaries[policy] = {
            "episodes": len(selected),
            "jobs": jobs,
            "completed_jobs": completed,
            "mean_episode_return": float(np.mean([item.total_return for item in selected])),
            "mean_makespan": float(np.mean([item.makespan for item in selected])),
            "throughput": completed / total_makespan if total_makespan > 0.0 else 0.0,
            "mean_wait": float(np.mean(waits)) if waits else 0.0,
            "p95_wait": percentile(waits, 95.0),
            "mean_slowdown": float(np.mean(slowdowns)) if slowdowns else 1.0,
            "p95_slowdown": percentile(slowdowns, 95.0),
            "sla_violations": violations,
            "sla_jobs": sla_jobs,
            "sla_violation_rate": violations / sla_jobs if sla_jobs else 0.0,
            "cpu_utilization": cpu_time / cpu_capacity_time if cpu_capacity_time else 0.0,
            "memory_utilization": (
                memory_time / memory_capacity_time if memory_capacity_time else 0.0
            ),
            "invalid_actions": sum(item.invalid_actions for item in selected),
            "decisions": sum(item.decisions for item in selected),
            "mean_decision_latency_us": float(np.mean(latencies)) if latencies else 0.0,
            "p95_decision_latency_us": percentile(latencies, 95.0),
        }
    return summaries


def percentage_improvement(candidate: float, reference: float, lower_is_better: bool) -> float | None:
    if abs(reference) < 1.0e-12:
        return None
    direction = reference - candidate if lower_is_better else candidate - reference
    return 100.0 * direction / abs(reference)


def compare_to_reference(
    summaries: dict[str, dict[str, float | int]], reference: str
) -> dict[str, float | int | None]:
    """Compare ONNX outcomes with one named baseline using interpretable deltas."""
    if reference not in summaries:
        raise ValueError(f"reference policy {reference!r} was not evaluated")
    learned = summaries["onnx"]
    baseline = summaries[reference]
    return {
        "reference": reference,
        "return_delta": float(learned["mean_episode_return"])
        - float(baseline["mean_episode_return"]),
        "mean_wait_reduction_percent": percentage_improvement(
            float(learned["mean_wait"]), float(baseline["mean_wait"]), True
        ),
        "p95_wait_reduction_percent": percentage_improvement(
            float(learned["p95_wait"]), float(baseline["p95_wait"]), True
        ),
        "makespan_reduction_percent": percentage_improvement(
            float(learned["mean_makespan"]), float(baseline["mean_makespan"]), True
        ),
        "throughput_improvement_percent": percentage_improvement(
            float(learned["throughput"]), float(baseline["throughput"]), False
        ),
        "sla_violation_rate_delta_points": 100.0
        * (
            float(learned["sla_violation_rate"])
            - float(baseline["sla_violation_rate"])
        ),
        "invalid_actions": int(learned["invalid_actions"]),
    }


def select_temporal_holdout(
    workload: np.ndarray, fraction: float, max_jobs: int | None
) -> np.ndarray:
    """Select the most recent fraction of a chronologically ordered trace."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("holdout fraction must be in (0, 1]")
    count = max(1, math.ceil(len(workload) * fraction))
    holdout = workload[-count:]
    if max_jobs is not None:
        if max_jobs <= 0:
            raise ValueError("max-jobs must be positive")
        holdout = holdout[-max_jobs:]
    return holdout.copy()


def write_reports(
    output: Path,
    results: Sequence[EpisodeResult],
    summaries: dict[str, dict[str, float | int]],
    comparison: dict[str, float | int | None],
    configuration: dict[str, object],
) -> None:
    """Atomically write summary JSON and sibling per-episode CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "configuration": configuration,
        "summary": summaries,
        "onnx_comparison": comparison,
        "episodes": [result.record() for result in results],
    }
    temporary = output.with_name(output.name + ".tmp")
    csv_output = output.with_suffix(".episodes.csv")
    temporary_csv = csv_output.with_name(csv_output.name + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        records = [result.record() for result in results]
        with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(output)
        temporary_csv.replace(csv_output)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_csv.unlink(missing_ok=True)


def print_summary(summaries: dict[str, dict[str, float | int]]) -> None:
    columns = (
        "policy", "return", "mean_wait", "p95_wait", "sla_rate", "throughput",
        "cpu_util", "mem_util", "p95_latency_us", "invalid",
    )
    print(" ".join(f"{column:>15}" for column in columns))
    for policy, values in sorted(summaries.items()):
        row = (
            policy,
            f"{float(values['mean_episode_return']):.4f}",
            f"{float(values['mean_wait']):.3f}",
            f"{float(values['p95_wait']):.3f}",
            f"{float(values['sla_violation_rate']):.3%}",
            f"{float(values['throughput']):.4f}",
            f"{float(values['cpu_utilization']):.3%}",
            f"{float(values['memory_utilization']):.3%}",
            f"{float(values['p95_decision_latency_us']):.2f}",
            str(values["invalid_actions"]),
        )
        print(" ".join(f"{value:>15}" for value in row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    parser.add_argument("--nodes", type=parse_nodes, default=parse_nodes("32:128,48:192,64:256"))
    parser.add_argument("--queue-capacity", type=int, default=32)
    parser.add_argument("--jobs-per-episode", type=int, default=2_000)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument(
        "--heuristics", nargs="+", choices=("fifo", "sjf", "drf", "best_fit"),
        default=["fifo", "sjf", "drf", "best_fit"],
    )
    parser.add_argument("--reference", choices=("fifo", "sjf", "drf", "best_fit"), default="sjf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--invalid-action-penalty", type=float, default=2.0)
    parser.add_argument("--time-scale", type=float, default=3600.0)
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
    if args.reference not in args.heuristics:
        raise ValueError("reference policy must be included in --heuristics")
    workload = load_trace(args.trace, args)
    holdout = select_temporal_holdout(workload, args.holdout_fraction, args.max_jobs)
    config = EvaluationConfig(
        node_capacities=args.nodes,
        queue_capacity=args.queue_capacity,
        jobs_per_episode=args.jobs_per_episode,
        seed=args.seed,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        invalid_action_penalty=args.invalid_action_penalty,
        time_scale=args.time_scale,
    )
    results = evaluate_workload(holdout, args.model, config, args.heuristics)
    summaries = aggregate_results(results)
    comparison = compare_to_reference(summaries, args.reference)
    report_config = {
        "trace": str(args.trace),
        "model": str(args.model),
        "holdout_fraction": args.holdout_fraction,
        "holdout_jobs": len(holdout),
        "node_capacities": args.nodes.tolist(),
        "queue_capacity": args.queue_capacity,
        "jobs_per_episode": args.jobs_per_episode,
        "heuristics": args.heuristics,
        "reference": args.reference,
        "reward": {
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "invalid_action_penalty": args.invalid_action_penalty,
            "time_scale": args.time_scale,
        },
    }
    write_reports(args.output, results, summaries, comparison, report_config)
    print_summary(summaries)
    print(f"\nONNX comparison vs {args.reference}: {json.dumps(comparison, sort_keys=True)}")
    print(f"reports: {args.output} and {args.output.with_suffix('.episodes.csv')}")


if __name__ == "__main__":
    main()
