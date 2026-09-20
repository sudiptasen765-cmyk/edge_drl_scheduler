"""
agents/ppo_config.py

Configuration for the masked-PPO training run (Phase 6). Mirrors the style
already used by RewardConfig (rewards/reward_engine.py) and EnvConfig
(environment/edge_scheduling_env.py): immutable dataclasses, validated on
construction, YAML-loadable, with unknown keys rejected so a typo in
config/ppo_config.yaml fails loudly at load time instead of silently
falling back to a default and quietly training with the wrong hyperparameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Sequence


def _build_dataclass(cls, data, label):
    """Construct a dataclass from a dict, rejecting unknown keys (typo guard)."""
    data = dict(data or {})
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(
            f"Unknown {label} key(s) {sorted(unknown)}. Valid: {sorted(valid)}"
        )
    return cls(**data)


@dataclass(frozen=True)
class PPOAlgoConfig:
    """Hyperparameters passed straight through to sb3_contrib.MaskablePPO."""

    policy: str = "MlpPolicy"
    net_arch: Sequence[int] = field(default_factory=lambda: (128, 128))
    learning_rate: float = 3.0e-4
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    normalize_advantage: bool = True

    def __post_init__(self):
        object.__setattr__(self, "net_arch", tuple(int(n) for n in self.net_arch))
        if not self.net_arch:
            raise ValueError("ppo.net_arch must be non-empty.")
        for name in (
            "learning_rate", "gamma", "gae_lambda", "clip_range",
            "vf_coef", "max_grad_norm",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"ppo.{name} must be > 0.")
        if self.n_steps <= 0 or self.batch_size <= 0 or self.n_epochs <= 0:
            raise ValueError("ppo.n_steps, batch_size and n_epochs must be > 0.")
        if self.ent_coef < 0:
            raise ValueError("ppo.ent_coef must be >= 0.")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("ppo.target_kl must be > 0 or null.")

    @classmethod
    def from_dict(cls, data: dict | None) -> "PPOAlgoConfig":
        return _build_dataclass(cls, data, "ppo config")


@dataclass(frozen=True)
class TrainingConfig:
    """Everything about the training run that isn't a PPO hyperparameter:
    how long to train, how many parallel envs, how often to checkpoint and
    evaluate, and where to write output."""

    total_timesteps: int = 300_000
    num_envs: int = 4
    seed: int = 42
    eval_freq_steps: int = 10_000
    eval_episodes: int = 5
    checkpoint_freq_steps: int = 25_000
    log_dir: str = "results/logs/ppo"
    checkpoint_dir: str = "models/ppo_checkpoints"
    tensorboard_log: str = "results/logs/tensorboard"

    def __post_init__(self):
        for name in (
            "total_timesteps", "num_envs", "eval_freq_steps",
            "eval_episodes", "checkpoint_freq_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"training.{name} must be > 0.")

    @classmethod
    def from_dict(cls, data: dict | None) -> "TrainingConfig":
        return _build_dataclass(cls, data, "training config")


@dataclass(frozen=True)
class PPOConfig:
    ppo: PPOAlgoConfig = field(default_factory=PPOAlgoConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PPOConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        unknown = set(data) - {"ppo", "training"}
        if unknown:
            raise ValueError(
                f"Unknown top-level key(s) in ppo config {sorted(unknown)}. "
                "Valid: ppo, training"
            )
        return cls(
            ppo=PPOAlgoConfig.from_dict(data.get("ppo")),
            training=TrainingConfig.from_dict(data.get("training")),
        )