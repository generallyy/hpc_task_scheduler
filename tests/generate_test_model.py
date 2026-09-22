#!/usr/bin/env python3
"""Generate a tiny masked policy used by the native inference smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_offline_rl import ExportedActor, MLP


def generate(output: Path, state_dim: int = 12, action_dim: int = 5) -> None:
    """Export a deterministic, verified policy with the production input contract."""
    torch.manual_seed(17)
    actor = MLP(state_dim, action_dim, hidden_dim=8, depth=1)
    model = ExportedActor(actor, torch.zeros(state_dim), torch.ones(state_dim)).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (
            torch.zeros(1, state_dim, dtype=torch.float32),
            torch.ones(1, action_dim, dtype=torch.uint8),
        ),
        output,
        input_names=["state", "action_mask"],
        output_names=["action_logits"],
        opset_version=18,
        dynamo=True,
        verify=True,
        external_data=False,
        verbose=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    generate(arguments.output)


if __name__ == "__main__":
    main()
