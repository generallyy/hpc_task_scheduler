#!/usr/bin/env python3
"""Train a discrete Conservative Q-Learning policy and export it to ONNX."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

LOGGER = logging.getLogger("offline_cql")


@dataclass(frozen=True)
class TrainConfig:
    """Serializable CQL optimization configuration."""

    hidden_dim: int = 256
    depth: int = 3
    batch_size: int = 512
    epochs: int = 100
    learning_rate: float = 3.0e-4
    discount: float = 0.99
    target_tau: float = 0.005
    cql_alpha: float = 1.0
    entropy_temperature: float = 0.1
    behavior_clone_weight: float = 0.1
    weight_decay: float = 1.0e-5
    validation_fraction: float = 0.1
    seed: int = 42


class MLP(nn.Module):
    """Layer-normalized MLP used by both actor and critics."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()))
            width = hidden_dim
        layers.append(nn.Linear(width, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class ExportedActor(nn.Module):
    """Actor plus observation normalization, suitable for standalone inference."""

    def __init__(self, actor: nn.Module, mean: Tensor, standard_deviation: Tensor) -> None:
        super().__init__()
        self.actor = actor
        self.register_buffer("observation_mean", mean.detach().clone())
        self.register_buffer("observation_std", standard_deviation.detach().clone())

    def forward(self, state: Tensor, action_mask: Tensor) -> Tensor:
        logits = self.actor((state - self.observation_mean) / self.observation_std)
        return torch.where(action_mask != 0, logits, torch.full_like(logits, -1.0e9))


def load_dataset(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and validate a collector NPZ/HDF5 file."""
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files if name != "metadata"}
            raw_metadata = source["metadata"].item() if "metadata" in source else "{}"
    elif path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as source:
            arrays = {name: source[name][...] for name in source.keys()}
            raw_metadata = source.attrs.get("metadata", "{}")
    else:
        raise ValueError("dataset extension must be .npz, .h5, or .hdf5")
    metadata = json.loads(raw_metadata)
    required = {
        "states", "actions", "rewards", "next_states", "dones",
        "action_masks", "next_action_masks",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"dataset is missing arrays: {', '.join(sorted(missing))}")
    count = arrays["states"].shape[0]
    if count == 0 or any(arrays[name].shape[0] != count for name in required):
        raise ValueError("transition arrays must be nonempty and have equal first dimensions")
    if arrays["states"].ndim != 2 or arrays["next_states"].ndim != 2:
        raise ValueError("states and next_states must be rank-2 arrays")
    if arrays["action_masks"].ndim != 2 or arrays["next_action_masks"].ndim != 2:
        raise ValueError("action masks must be rank-2 arrays")
    if any(arrays[name].ndim != 1 for name in ("actions", "rewards", "dones")):
        raise ValueError("actions, rewards, and dones must be rank-1 arrays")
    if "episode_ids" in arrays and (
        arrays["episode_ids"].ndim != 1 or arrays["episode_ids"].shape[0] != count
    ):
        raise ValueError("episode_ids must be a rank-1 array aligned with transitions")
    for name in ("states", "next_states", "rewards"):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"dataset array {name} contains NaN or infinity")
    if not np.all(np.equal(arrays["actions"], np.floor(arrays["actions"]))):
        raise ValueError("actions must be integer valued")
    if not np.all(np.isin(arrays["dones"], (0, 1, False, True))):
        raise ValueError("dones must contain only boolean/0/1 values")
    return arrays, metadata


def masked_distribution(logits: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Return masked probabilities and numerically safe log probabilities."""
    safe_mask = mask.bool()
    masked_logits = logits.masked_fill(~safe_mask, torch.finfo(logits.dtype).min)
    log_probabilities = F.log_softmax(masked_logits, dim=-1)
    probabilities = log_probabilities.exp()
    return probabilities, log_probabilities


@torch.no_grad()
def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
        target_parameter.lerp_(source_parameter, tau)


@torch.no_grad()
def evaluate(
    loader: DataLoader[tuple[Tensor, ...]],
    actor: nn.Module,
    critic_one: nn.Module,
    critic_two: nn.Module,
    mean: Tensor,
    std: Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Compare policy actions and Q estimates with logged heuristic decisions."""
    actor.eval()
    critic_one.eval()
    critic_two.eval()
    agreements: list[Tensor] = []
    policy_values: list[Tensor] = []
    behavior_values: list[Tensor] = []
    for batch in loader:
        states, actions, _, _, _, masks, _ = [value.to(device) for value in batch]
        normalized = (states - mean) / std
        logits = actor(normalized).masked_fill(~masks.bool(), -torch.inf)
        selected = logits.argmax(dim=-1)
        q_values = torch.minimum(critic_one(normalized), critic_two(normalized))
        agreements.append((selected == actions).float().cpu())
        policy_values.append(q_values.gather(1, selected.unsqueeze(1)).squeeze(1).cpu())
        behavior_values.append(q_values.gather(1, actions.unsqueeze(1)).squeeze(1).cpu())
    policy_q = torch.cat(policy_values).mean().item()
    behavior_q = torch.cat(behavior_values).mean().item()
    return {
        "heuristic_action_agreement": torch.cat(agreements).mean().item(),
        "policy_q": policy_q,
        "heuristic_q": behavior_q,
        "estimated_q_improvement": policy_q - behavior_q,
    }


def make_loaders(
    arrays: dict[str, np.ndarray], config: TrainConfig
) -> tuple[DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]], Tensor, Tensor]:
    states = torch.as_tensor(arrays["states"], dtype=torch.float32)
    next_states = torch.as_tensor(arrays["next_states"], dtype=torch.float32)
    dataset = TensorDataset(
        states,
        torch.as_tensor(arrays["actions"], dtype=torch.long),
        torch.as_tensor(arrays["rewards"], dtype=torch.float32),
        next_states,
        torch.as_tensor(arrays["dones"], dtype=torch.float32),
        torch.as_tensor(arrays["action_masks"], dtype=torch.bool),
        torch.as_tensor(arrays["next_action_masks"], dtype=torch.bool),
    )
    if len(dataset) < 2:
        raise ValueError("dataset requires at least two transitions")
    random_generator = np.random.default_rng(config.seed)
    episode_ids = arrays.get("episode_ids")
    if episode_ids is not None and len(np.unique(episode_ids)) >= 2:
        episodes = np.unique(episode_ids)
        random_generator.shuffle(episodes)
        validation_episode_count = min(
            len(episodes) - 1, max(1, int(len(episodes) * config.validation_fraction))
        )
        validation_selector = np.isin(episode_ids, episodes[:validation_episode_count])
        validation_indices = np.flatnonzero(validation_selector).tolist()
        training_indices = np.flatnonzero(~validation_selector).tolist()
    else:
        indices = random_generator.permutation(len(dataset))
        validation_size = min(
            len(dataset) - 1, max(1, int(len(dataset) * config.validation_fraction))
        )
        validation_indices = indices[:validation_size].tolist()
        training_indices = indices[validation_size:].tolist()
    training = Subset(dataset, training_indices)
    validation = Subset(dataset, validation_indices)
    train_loader = DataLoader(
        training, batch_size=config.batch_size, shuffle=True, pin_memory=torch.cuda.is_available()
    )
    validation_loader = DataLoader(validation, batch_size=config.batch_size, shuffle=False)
    normalization_values = torch.cat(
        (states[training_indices], next_states[training_indices]), dim=0
    )
    mean = normalization_values.mean(dim=0)
    std = normalization_values.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    return train_loader, validation_loader, mean, std


def train(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    config: TrainConfig,
    output_dir: Path,
    device: torch.device,
) -> Path:
    """Optimize twin discrete CQL critics and an entropy-regularized actor."""
    state_dim = int(arrays["states"].shape[1])
    action_dim = int(arrays["action_masks"].shape[1])
    if arrays["next_states"].shape[1] != state_dim:
        raise ValueError("states and next_states have inconsistent widths")
    if arrays["action_masks"].shape != (arrays["states"].shape[0], action_dim) or arrays[
        "next_action_masks"
    ].shape != (arrays["states"].shape[0], action_dim):
        raise ValueError("action mask arrays have inconsistent shapes")
    if np.any(arrays["actions"] < 0) or np.any(arrays["actions"] >= action_dim):
        raise ValueError("dataset contains out-of-range actions")
    row_indices = np.arange(arrays["actions"].shape[0])
    if not np.all(arrays["action_masks"][row_indices, arrays["actions"]]):
        raise ValueError("dataset contains a logged action marked invalid by its action mask")

    train_loader, validation_loader, mean, std = make_loaders(arrays, config)
    mean, std = mean.to(device), std.to(device)
    actor = MLP(state_dim, action_dim, config.hidden_dim, config.depth).to(device)
    critic_one = MLP(state_dim, action_dim, config.hidden_dim, config.depth).to(device)
    critic_two = MLP(state_dim, action_dim, config.hidden_dim, config.depth).to(device)
    target_one = MLP(state_dim, action_dim, config.hidden_dim, config.depth).to(device)
    target_two = MLP(state_dim, action_dim, config.hidden_dim, config.depth).to(device)
    target_one.load_state_dict(critic_one.state_dict())
    target_two.load_state_dict(critic_two.state_dict())
    critic_optimizer = torch.optim.AdamW(
        list(critic_one.parameters()) + list(critic_two.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    best_score = -float("inf")
    best_checkpoint = output_dir / "best_policy.pt"

    for epoch in range(1, config.epochs + 1):
        actor.train()
        critic_one.train()
        critic_two.train()
        loss_totals = np.zeros(3, dtype=np.float64)
        batches = 0
        for batch in train_loader:
            states, actions, rewards, next_states, dones, masks, next_masks = [
                value.to(device, non_blocking=True) for value in batch
            ]
            normalized_states = (states - mean) / std
            normalized_next_states = (next_states - mean) / std

            with torch.no_grad():
                next_probabilities, next_log_probabilities = masked_distribution(
                    actor(normalized_next_states), next_masks
                )
                next_q = torch.minimum(
                    target_one(normalized_next_states), target_two(normalized_next_states)
                )
                next_soft_value = (
                    next_probabilities
                    * (next_q - config.entropy_temperature * next_log_probabilities)
                ).sum(dim=1)
                target = rewards + config.discount * (1.0 - dones) * next_soft_value

            q_one = critic_one(normalized_states)
            q_two = critic_two(normalized_states)
            data_q_one = q_one.gather(1, actions.unsqueeze(1)).squeeze(1)
            data_q_two = q_two.gather(1, actions.unsqueeze(1)).squeeze(1)
            bellman_loss = F.mse_loss(data_q_one, target) + F.mse_loss(data_q_two, target)

            # Discrete CQL regularizer:
            # alpha * E_s[log sum_a exp(Q(s,a)) - Q(s,a_dataset)].
            # It lowers unseen-action values while preserving logged-action values.
            masked_q_one = q_one.masked_fill(~masks, torch.finfo(q_one.dtype).min)
            masked_q_two = q_two.masked_fill(~masks, torch.finfo(q_two.dtype).min)
            conservative_loss = (
                torch.logsumexp(masked_q_one, dim=1).mean() - data_q_one.mean()
                + torch.logsumexp(masked_q_two, dim=1).mean() - data_q_two.mean()
            )
            critic_loss = bellman_loss + config.cql_alpha * conservative_loss
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                list(critic_one.parameters()) + list(critic_two.parameters()), 10.0
            )
            critic_optimizer.step()

            probabilities, log_probabilities = masked_distribution(
                actor(normalized_states), masks
            )
            with torch.no_grad():
                conservative_q = torch.minimum(
                    critic_one(normalized_states), critic_two(normalized_states)
                )
            # J_pi = E_s sum_a pi(a|s)[temperature*log pi(a|s) - min(Q1,Q2)]
            policy_loss = (
                probabilities
                * (config.entropy_temperature * log_probabilities - conservative_q)
            ).sum(dim=1).mean()
            behavior_clone_loss = F.cross_entropy(
                actor(normalized_states).masked_fill(~masks, -torch.inf), actions
            )
            actor_loss = policy_loss + config.behavior_clone_weight * behavior_clone_loss
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            actor_optimizer.step()
            soft_update(critic_one, target_one, config.target_tau)
            soft_update(critic_two, target_two, config.target_tau)
            loss_totals += [critic_loss.item(), actor_loss.item(), conservative_loss.item()]
            batches += 1

        validation_metrics = evaluate(
            validation_loader, actor, critic_one, critic_two, mean, std, device
        )
        baseline_returns = arrays.get("episode_returns")
        baseline_mean_return = (
            float(np.asarray(baseline_returns).mean())
            if baseline_returns is not None and np.asarray(baseline_returns).size > 0
            else None
        )
        record = {
            "epoch": epoch,
            "critic_loss": float(loss_totals[0] / batches),
            "actor_loss": float(loss_totals[1] / batches),
            "cql_penalty": float(loss_totals[2] / batches),
            **validation_metrics,
            "baseline_mean_episode_return": baseline_mean_return,
        }
        with metrics_path.open("w" if epoch == 1 else "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=True) + "\n")
        LOGGER.info("epoch=%d metrics=%s", epoch, json.dumps(record))
        score = validation_metrics["policy_q"]
        if not np.isfinite(score):
            raise FloatingPointError("non-finite validation Q value; lower the learning rate")
        if epoch == 1 or score > best_score:
            best_score = score
            checkpoint_payload = {
                "actor_state_dict": actor.state_dict(),
                "critic_one_state_dict": critic_one.state_dict(),
                "critic_two_state_dict": critic_two.state_dict(),
                "observation_mean": mean.cpu(),
                "observation_std": std.cpu(),
                "state_dim": state_dim,
                "action_dim": action_dim,
                "config": asdict(config),
                "dataset_metadata": metadata,
                "epoch": epoch,
                "validation_metrics": validation_metrics,
            }
            temporary_checkpoint = best_checkpoint.with_suffix(".pt.tmp")
            try:
                torch.save(checkpoint_payload, temporary_checkpoint)
                temporary_checkpoint.replace(best_checkpoint)
            finally:
                temporary_checkpoint.unlink(missing_ok=True)

    checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
    export_actor = MLP(state_dim, action_dim, config.hidden_dim, config.depth)
    export_actor.load_state_dict(checkpoint["actor_state_dict"])
    model = ExportedActor(
        export_actor, checkpoint["observation_mean"], checkpoint["observation_std"]
    ).eval()
    onnx_path = output_dir / "scheduler_policy.onnx"
    temporary_onnx_path = output_dir / "scheduler_policy.onnx.tmp"
    batch_dimension = torch.export.Dim("batch", min=1)
    try:
        torch.onnx.export(
            model,
            (
                torch.zeros(1, state_dim, dtype=torch.float32),
                torch.ones(1, action_dim, dtype=torch.uint8),
            ),
            temporary_onnx_path,
            input_names=["state", "action_mask"],
            output_names=["action_logits"],
            dynamic_shapes=({0: batch_dimension}, {0: batch_dimension}),
            opset_version=18,
            dynamo=True,
            verify=True,
            external_data=False,
            verbose=False,
        )
        temporary_onnx_path.replace(onnx_path)
    finally:
        temporary_onnx_path.unlink(missing_ok=True)
    LOGGER.info("exported best checkpoint to %s", onnx_path)
    return onnx_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    for field_name, field_definition in TrainConfig.__dataclass_fields__.items():
        default = field_definition.default
        option = "--" + field_name.replace("_", "-")
        parser.add_argument(option, type=type(default), default=default)
    return parser


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if value == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise ValueError("MPS was requested but is not available")
    return torch.device(value)


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_values = {
        name: getattr(args, name) for name in TrainConfig.__dataclass_fields__
    }
    config = TrainConfig(**config_values)
    if config.epochs <= 0 or config.batch_size <= 0 or config.hidden_dim <= 0 or config.depth <= 0:
        raise ValueError("epochs, batch-size, hidden-dim, and depth must be positive")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be strictly between zero and one")
    if not 0.0 <= config.discount <= 1.0 or not 0.0 < config.target_tau <= 1.0:
        raise ValueError("discount must be in [0,1] and target-tau in (0,1]")
    if config.learning_rate <= 0.0 or config.cql_alpha < 0.0 or config.entropy_temperature < 0.0:
        raise ValueError("learning-rate must be positive; CQL alpha and entropy must be nonnegative")
    if config.behavior_clone_weight < 0.0 or config.weight_decay < 0.0:
        raise ValueError("behavior-clone-weight and weight-decay must be nonnegative")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    arrays, metadata = load_dataset(args.dataset)
    device = resolve_device(args.device)
    LOGGER.info("training on %s with %d transitions", device, arrays["states"].shape[0])
    train(arrays, metadata, config, args.output_dir, device)


if __name__ == "__main__":
    main()
