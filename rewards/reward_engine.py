"""
reward_engine.py

Configurable multi-objective reward for the edge task-scheduling DRL agent.

This module is deliberately simulator-agnostic: it never imports anything from
`environment/`. The Gymnasium environment translates what happened in one step
(tasks completed, tasks failed, server loads, energy used) into a plain
`RewardInputs` object, and `RewardEngine.compute()` turns that into a scalar
reward plus a per-term breakdown. That separation gives us:

  * unit tests that need no simulator (see test_drl_reward.py),
  * one-line ablations (Phase 10): set a term's weight to 0 in
    config/reward_config.yaml, or call `RewardConfig.without("energy")`,
  * a per-term breakdown for logging and explainability.

Reward terms
------------
Five costs (subtracted) and one benefit (added). Every term is first computed
as a NON-NEGATIVE, roughly unit-scaled "raw" value; the weight then decides
its influence. Raw values are always reported, even when a weight is 0, so an
ablated term can still be monitored.

  latency      Sum over tasks that COMPLETED this step of
               min(latency_ms / latency_ref_ms, latency_cap).
               (Realized latency: arrival -> completion.)
  sla          Priority-weighted count of deadline violations this step:
               tasks that finished late, plus tasks killed for missing their
               deadline while running.
  rejection    Priority-weighted count of tasks that were never served this
               step (server full, queue overflow in transit, invalid server).
               Kept separate from `sla` on purpose: if someone ablates the
               SLA term, dropping tasks must not become free.
  balance      Load imbalance across servers right after the decision:
               min(1, 2 * std(load_scores)). 0 = perfectly even.
  energy       Energy consumed by the whole fleet since the previous
               decision, in joules, divided by `energy_ref_j`.
  utilization  (benefit) Capacity-weighted mean CPU utilization of the fleet
               at the end of the step, in [0, 1].

Timing note: latency / sla / rejection / energy are *event or integral*
terms (each completion, failure and joule is counted exactly once over an
episode, credited at the step during which it happened). balance and
utilization are *sampled* per decision. This is standard for
decision-event RL schedulers, and the env tests verify that no event is ever
counted twice or dropped.

Sign convention: `raw` values are magnitudes >= 0. `weighted` values are the
signed contribution to the total (negative for costs, positive for the
utilization benefit). total = reward_scale * sum(weighted.values()).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

# All terms, in a fixed order (used for logging and ablation loops).
TERMS: tuple[str, ...] = (
    "latency",
    "sla",
    "rejection",
    "balance",
    "energy",
    "utilization",
)

# Terms that ADD to the reward (everything else is subtracted).
BENEFIT_TERMS: frozenset[str] = frozenset({"utilization"})

# A failure with one of these reasons is a deadline violation (-> "sla" term).
# Every other failure reason means the task was never served (-> "rejection").
DEADLINE_FAILURE_REASONS: frozenset[str] = frozenset({"deadline_missed_while_running"})

_DEFAULT_WEIGHTS = {
    "latency": 0.5,
    "sla": 3.0,
    "rejection": 4.0,
    "balance": 0.3,
    "energy": 0.2,
    "utilization": 0.1,
}
_DEFAULT_PRIORITY_WEIGHTS = {"high": 2.0, "medium": 1.0, "low": 0.5}


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class RewardConfig:
    """Immutable reward configuration. Build it from YAML with `from_yaml`."""

    weights: Mapping[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    latency_ref_ms: float = 1000.0
    latency_cap: float = 3.0
    energy_ref_j: float = 100.0
    priority_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_PRIORITY_WEIGHTS)
    )
    # How per-server "load" is scored for the balance term and the env's
    # load-aware features. Same formula as GreedyScheduler by default.
    load_cpu_weight: float = 0.5
    load_queue_weight: float = 0.5
    reward_scale: float = 1.0

    def __post_init__(self):
        # Merge user weights over defaults, and reject typos loudly: a
        # misspelled weight key silently doing nothing would ruin an ablation.
        weights = dict(_DEFAULT_WEIGHTS)
        unknown = set(self.weights) - set(TERMS)
        if unknown:
            raise ValueError(
                f"Unknown reward weight key(s) {sorted(unknown)}. Valid: {list(TERMS)}"
            )
        weights.update({k: float(v) for k, v in self.weights.items()})
        if any(v < 0 for v in weights.values()):
            raise ValueError(
                "Reward weights must be >= 0 (sign is fixed per term: costs are "
                "subtracted, utilization is added)."
            )
        object.__setattr__(self, "weights", weights)

        priority_weights = {k: float(v) for k, v in self.priority_weights.items()}
        if any(v < 0 for v in priority_weights.values()):
            raise ValueError("priority_weights must be >= 0.")
        object.__setattr__(self, "priority_weights", priority_weights)

        for name in ("latency_ref_ms", "latency_cap", "energy_ref_j", "reward_scale"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0 (got {getattr(self, name)}).")
        if self.load_cpu_weight < 0 or self.load_queue_weight < 0:
            raise ValueError("load_cpu_weight / load_queue_weight must be >= 0.")

    # -- construction helpers -------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "RewardConfig":
        data = dict(data or {})
        valid = {
            "weights", "latency_ref_ms", "latency_cap", "energy_ref_j",
            "priority_weights", "load_cpu_weight", "load_queue_weight", "reward_scale",
        }
        unknown = set(data) - valid
        if unknown:
            raise ValueError(
                f"Unknown reward config key(s) {sorted(unknown)}. Valid: {sorted(valid)}"
            )
        return cls(**data)

    @classmethod
    def from_yaml(
        cls, path: str | Path, overrides: Mapping | None = None
    ) -> "RewardConfig":
        """
        Load the `reward:` section of a YAML file. `overrides` is merged on top
        (one level deep for `weights` / `priority_weights`), e.g. for ablations:
            RewardConfig.from_yaml(p, overrides={"weights": {"energy": 0.0}})
        """
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        data = copy.deepcopy(raw.get("reward", {}))
        for key, value in (overrides or {}).items():
            if isinstance(value, Mapping) and isinstance(data.get(key), Mapping):
                data[key] = {**data[key], **value}
            else:
                data[key] = value
        return cls.from_dict(data)

    # -- ablation helpers -----------------------------------------------

    def with_weights(self, **weights: float) -> "RewardConfig":
        """Copy with some weights replaced, e.g. cfg.with_weights(energy=0.5)."""
        unknown = set(weights) - set(TERMS)
        if unknown:
            raise ValueError(f"Unknown term(s) {sorted(unknown)}. Valid: {list(TERMS)}")
        return RewardConfig(
            weights={**self.weights, **weights},
            latency_ref_ms=self.latency_ref_ms,
            latency_cap=self.latency_cap,
            energy_ref_j=self.energy_ref_j,
            priority_weights=dict(self.priority_weights),
            load_cpu_weight=self.load_cpu_weight,
            load_queue_weight=self.load_queue_weight,
            reward_scale=self.reward_scale,
        )

    def without(self, *terms: str) -> "RewardConfig":
        """Copy with the given terms switched off (weight 0)."""
        return self.with_weights(**{t: 0.0 for t in terms})


# ----------------------------------------------------------------------
# Inputs / outputs
# ----------------------------------------------------------------------

@dataclass
class RewardInputs:
    """
    Everything that happened during one environment step, as plain data.

    completed: one mapping per task that finished this step, with keys
               'latency_ms' (float), 'met_deadline' (bool), 'priority' (str).
    failed:    one mapping per task that failed this step, with keys
               'reason' (str), 'priority' (str).
    load_scores:      per-server load in [0, ~1], measured right after the
                      scheduling decision (used by the balance term).
    cpu_utilization:  per-server CPU utilization in [0, 1] at end of step.
    cpu_cores:        per-server core counts (weights for the utilization mean).
    energy_delta_j:   fleet energy consumed since the previous step (>= 0).
    """

    completed: Sequence[Mapping] = ()
    failed: Sequence[Mapping] = ()
    load_scores: Sequence[float] = ()
    cpu_utilization: Sequence[float] = ()
    cpu_cores: Sequence[float] = ()
    energy_delta_j: float = 0.0


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    raw: dict[str, float]
    weighted: dict[str, float]
    counts: dict[str, int]


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------

class RewardEngine:
    """Stateless: the same inputs always give the same reward."""

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()

    def _priority_weight(self, priority: str) -> float:
        try:
            return self.config.priority_weights[str(priority)]
        except KeyError:
            raise KeyError(
                f"Unknown task priority '{priority}'. "
                f"Configured priorities: {sorted(self.config.priority_weights)}"
            ) from None

    def compute(self, inputs: RewardInputs) -> RewardBreakdown:
        cfg = self.config

        # --- latency: realized latency of tasks completed this step ---
        latency = sum(
            min(float(c["latency_ms"]) / cfg.latency_ref_ms, cfg.latency_cap)
            for c in inputs.completed
        )

        # --- sla vs rejection: split failures, add late completions to sla ---
        late = [c for c in inputs.completed if not c["met_deadline"]]
        deadline_failed = [
            f for f in inputs.failed if f["reason"] in DEADLINE_FAILURE_REASONS
        ]
        rejected = [
            f for f in inputs.failed if f["reason"] not in DEADLINE_FAILURE_REASONS
        ]
        sla = sum(self._priority_weight(t["priority"]) for t in late) + sum(
            self._priority_weight(t["priority"]) for t in deadline_failed
        )
        rejection = sum(self._priority_weight(t["priority"]) for t in rejected)

        # --- balance: spread of per-server load right after the decision ---
        balance = self._imbalance(inputs.load_scores)

        # --- energy: joules since last step, normalised ---
        if inputs.energy_delta_j < 0:
            raise ValueError(
                f"energy_delta_j must be >= 0 (got {inputs.energy_delta_j}); "
                "energy counters should never go backwards."
            )
        energy = float(inputs.energy_delta_j) / cfg.energy_ref_j

        # --- utilization: capacity-weighted mean CPU utilization ---
        utilization = self._utilization(inputs.cpu_utilization, inputs.cpu_cores)

        raw = {
            "latency": float(latency),
            "sla": float(sla),
            "rejection": float(rejection),
            "balance": float(balance),
            "energy": float(energy),
            "utilization": float(utilization),
        }
        weighted = {
            term: (1.0 if term in BENEFIT_TERMS else -1.0) * cfg.weights[term] * raw[term]
            for term in TERMS
        }
        total = cfg.reward_scale * sum(weighted.values())
        counts = {
            "n_completed": len(inputs.completed),
            "n_late_completions": len(late),
            "n_deadline_failures": len(deadline_failed),
            "n_rejected": len(rejected),
        }
        return RewardBreakdown(total=float(total), raw=raw, weighted=weighted, counts=counts)

    # -- term helpers ---------------------------------------------------

    @staticmethod
    def _imbalance(load_scores: Sequence[float]) -> float:
        if len(load_scores) < 2:
            return 0.0
        return float(min(1.0, 2.0 * np.std(np.asarray(load_scores, dtype=float))))

    @staticmethod
    def _utilization(utils: Sequence[float], cores: Sequence[float]) -> float:
        if len(utils) == 0:
            return 0.0
        if len(utils) != len(cores):
            raise ValueError("cpu_utilization and cpu_cores must have the same length.")
        total_cores = float(np.sum(cores))
        if total_cores <= 0:
            return 0.0
        used = float(np.dot(np.asarray(utils, dtype=float), np.asarray(cores, dtype=float)))
        return float(np.clip(used / total_cores, 0.0, 1.0))