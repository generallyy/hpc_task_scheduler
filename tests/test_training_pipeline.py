"""End-to-end smoke test for CQL training, checkpointing, and ONNX inference."""

from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort
import pytest
import torch

from data_collector import CollectorConfig, collect
from evaluate_policy import (
    EvaluationConfig,
    aggregate_results,
    compare_to_reference,
    evaluate_workload,
    write_reports,
)
from train_offline_rl import TrainConfig, train

pytestmark = pytest.mark.filterwarnings(
    "ignore:# The axis name.*:UserWarning"
)


def test_one_epoch_training_exports_verified_masked_onnx(tmp_path) -> None:
    workload = np.asarray(
        [
            [0.0, 1.0, 1.0, 1.0, 3.0],
            [0.0, 2.0, 2.0, 2.0, 3.0],
            [1.0, 1.0, 2.0, 1.0, 3.0],
            [0.0, 1.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 1.0, 1.0, 3.0],
            [2.0, 1.0, 1.0, 1.0, 3.0],
        ],
        dtype=np.float64,
    )
    arrays = collect(
        workload,
        CollectorConfig(
            queue_capacity=2,
            node_capacities=np.asarray([[4.0, 8.0]], dtype=np.float32),
            heuristic="sjf",
            jobs_per_episode=3,
            seed=3,
        ),
    )
    configuration = TrainConfig(
        hidden_dim=16,
        depth=1,
        batch_size=4,
        epochs=1,
        learning_rate=1.0e-3,
        validation_fraction=0.5,
        seed=3,
    )
    output_directory = tmp_path / "artifacts"
    model_path = train(
        arrays,
        {"schema_version": 1, "heuristic": "sjf"},
        configuration,
        output_directory,
        torch.device("cpu"),
    )
    assert model_path.is_file()
    assert (output_directory / "best_policy.pt").is_file()
    metrics = [
        json.loads(line)
        for line in (output_directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(metrics) == 1
    assert np.isfinite(metrics[0]["critic_loss"])
    assert np.isfinite(metrics[0]["policy_q"])

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    state = arrays["states"][:1]
    mask = arrays["action_masks"][:1].astype(np.uint8)
    mask[0, 0] = 0
    logits = session.run(["action_logits"], {"state": state, "action_mask": mask})[0]
    assert logits.shape == mask.shape
    assert logits[0, 0] <= -1.0e8
    assert mask[0, int(logits.argmax(axis=1)[0])] == 1

    batched_logits = session.run(
        ["action_logits"],
        {
            "state": np.repeat(state, 2, axis=0),
            "action_mask": np.repeat(mask, 2, axis=0),
        },
    )[0]
    assert batched_logits.shape == (2, mask.shape[1])

    replay_results = evaluate_workload(
        workload,
        model_path,
        EvaluationConfig(
            node_capacities=np.asarray([[4.0, 8.0]], dtype=np.float32),
            queue_capacity=2,
            jobs_per_episode=3,
            seed=3,
            time_scale=3600.0,
        ),
    )
    assert len(replay_results) == 10  # Five policies over two identical episodes.
    assert all(result.completed_jobs == result.jobs for result in replay_results)
    assert all(len(result.wait_times) == result.jobs for result in replay_results)
    summaries = aggregate_results(replay_results)
    assert set(summaries) == {"best_fit", "drf", "fifo", "onnx", "sjf"}
    assert all(summary["invalid_actions"] == 0 for summary in summaries.values())
    assert all(0.0 <= summary["cpu_utilization"] <= 1.0 for summary in summaries.values())
    comparison = compare_to_reference(summaries, "sjf")
    assert comparison["reference"] == "sjf"
    report = tmp_path / "evaluation.json"
    write_reports(report, replay_results, summaries, comparison, {"test": True})
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["summary"] == summaries
    assert report.with_suffix(".episodes.csv").is_file()
