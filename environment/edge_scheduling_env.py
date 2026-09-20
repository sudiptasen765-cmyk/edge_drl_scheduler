"""
edge_scheduling_env.py

Gymnasium environment for DRL task scheduling on the heterogeneous edge fleet.

It WRAPS the existing simulator (`EdgeEnvironment`) and adds nothing to it: the
simulator stays scheduler-agnostic and unchanged. The wrapper only uses the
low-level interface the simulator already exposes for this purpose:
next_pending_task() / submit_decision() / advance_time().

Decision structure
------------------
One Gymnasium step == one scheduling decision for one task.

    reset()   fast-forwards the simulation to the first task that needs a
              decision and returns its observation.
    step(a)   sends the pending task to server `server_ids[a]`. If more tasks
              are already waiting in the same tick, the next one is returned
              immediately (no time passes). Otherwise the simulation advances,
              tick by tick, until the next task needs a decision or the
              episode ends.

This ordering is identical to `EdgeEnvironment.run_episode()`, so a baseline
scheduler driven through this environment produces exactly the same result as
when driven through run_episode() (tested in test_drl_env.py). That is what
makes baseline-vs-DRL comparisons fair.

Episode end: `terminated=True` when every task has been either completed or
failed and the fleet has drained. `truncated=True` only if the safety limit
`max_ticks` is hit.

Action space:  Discrete(num_servers); action i -> server_ids[i] (ascending id).
Action mask:   env.action_masks() and info["action_mask"] (bool array). The
               environment does NOT force the agent to obey it: an
               un-masked agent may pick an unavailable server and will
               simply be penalised (this is what the masking ablation needs).

Seeding: reset(seed=s) fixes the environment RNG, which fixes the sampled
workload, network scenario and task stream. Same seed -> identical episode.
reset(options={...}) may override: workload_type, network_scenario,
duration_ms, task_seed.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.edge_environment import EdgeEnvironment
from environment.network import NETWORK_SCENARIOS
from environment.state_builder import MaskConfig, ObservationConfig, StateBuilder
from environment.task_generator import WORKLOAD_PRESETS
from rewards.reward_engine import (
    DEADLINE_FAILURE_REASONS,
    TERMS,
    RewardConfig,
    RewardEngine,
    RewardInputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_CONFIG = PROJECT_ROOT / "config" / "server_config.yaml"
DEFAULT_ENV_CONFIG = PROJECT_ROOT / "config" / "env_config.yaml"
DEFAULT_REWARD_CONFIG = PROJECT_ROOT / "config" / "reward_config.yaml"

_RESET_OPTION_KEYS = {"workload_type", "network_scenario", "duration_ms", "task_seed"}


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class EnvConfig:
    num_users: int = 15
    time_step_ms: float = 10.0
    duration_ms: float = 10000.0
    workloads: tuple = ("normal", "heavy", "burst", "variable")
    network_scenarios: tuple = ("normal", "congested")
    allow_unseen: bool = False
    max_ticks: int = 60000
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    action_mask: MaskConfig = field(default_factory=MaskConfig)

    def __post_init__(self):
        object.__setattr__(self, "workloads", tuple(self.workloads))
        object.__setattr__(self, "network_scenarios", tuple(self.network_scenarios))
        if not self.workloads or not self.network_scenarios:
            raise ValueError("workloads and network_scenarios must be non-empty.")
        if self.time_step_ms <= 0 or self.duration_ms <= 0 or self.max_ticks <= 0:
            raise ValueError("time_step_ms, duration_ms and max_ticks must be > 0.")

    @classmethod
    def from_dict(cls, data: dict | None) -> "EnvConfig":
        """`data` has optional sections: environment, observation, action_mask."""
        data = copy.deepcopy(data or {})
        unknown_sections = set(data) - {"environment", "observation", "action_mask"}
        if unknown_sections:
            raise ValueError(
                f"Unknown env config section(s) {sorted(unknown_sections)}. "
                "Valid: environment, observation, action_mask"
            )
        env_section = dict(data.get("environment", {}))
        valid = {f.name for f in fields(cls)} - {"observation", "action_mask"}
        unknown = set(env_section) - valid
        if unknown:
            raise ValueError(
                f"Unknown environment config key(s) {sorted(unknown)}. Valid: {sorted(valid)}"
            )
        return cls(
            observation=ObservationConfig.from_dict(data.get("observation")),
            action_mask=MaskConfig.from_dict(data.get("action_mask")),
            **env_section,
        )

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict | None = None) -> "EnvConfig":
        """Load a YAML file; `overrides` is merged per section (one level deep)."""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for section, values in (overrides or {}).items():
            if not isinstance(values, dict):
                raise ValueError(
                    "env overrides must be section-level dicts, e.g. "
                    "{'environment': {'max_ticks': 5}} or "
                    "{'observation': {'include_feasibility': True}}; "
                    f"got {section!r}: {values!r}"
                )
            data[section] = {**data.get(section, {}), **values}
        return cls.from_dict(data)


def _load_server_configs(path: str | Path) -> list[dict]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["servers"]


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------

class EdgeSchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        server_configs: list[dict],
        env_config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
    ):
        super().__init__()
        self.cfg = env_config or EnvConfig()
        self.reward_config = reward_config or RewardConfig()
        self.reward_engine = RewardEngine(self.reward_config)

        self._validate_choices(self.cfg.workloads, self.cfg.network_scenarios)

        self.sim = EdgeEnvironment(
            server_configs, num_users=self.cfg.num_users, time_step_ms=self.cfg.time_step_ms
        )
        self.state_builder = StateBuilder(self.sim, self.cfg.observation, self.cfg.action_mask)
        self.server_ids: list[int] = self.state_builder.server_ids
        self._cpu_cores = [self.sim.servers[sid].cpu_cores for sid in self.server_ids]

        self.action_space = spaces.Discrete(len(self.server_ids))
        self.observation_space = spaces.Box(
            low=self.state_builder.low, high=self.state_builder.high, dtype=np.float32
        )

        # Per-episode state (real values are set in reset()).
        self._task = None
        self._truncated_at_reset = False
        self._energy_mark = 0.0
        self._mask = np.ones(len(self.server_ids), dtype=bool)
        self._mask_level = 3
        self._needs_reset = True
        self._ticks = 0
        self._episode_meta: dict = {}
        self._reset_accumulators()

    @classmethod
    def from_config_files(
        cls,
        server_config_path: str | Path | None = None,
        env_config_path: str | Path | None = None,
        reward_config_path: str | Path | None = None,
        env_overrides: dict | None = None,
        reward_overrides: dict | None = None,
    ) -> "EdgeSchedulingEnv":
        """
        Build from the YAML files in config/. Overrides make ablations and
        experiment variants one-liners, e.g.
            EdgeSchedulingEnv.from_config_files(
                reward_overrides={"weights": {"energy": 0.0}},
                env_overrides={"action_mask": {"overload_fraction": 0.8}})
        """
        return cls(
            server_configs=_load_server_configs(server_config_path or DEFAULT_SERVER_CONFIG),
            env_config=EnvConfig.from_yaml(env_config_path or DEFAULT_ENV_CONFIG, env_overrides),
            reward_config=RewardConfig.from_yaml(
                reward_config_path or DEFAULT_REWARD_CONFIG, reward_overrides
            ),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_choices(self, workloads, scenarios):
        for w in workloads:
            if w not in WORKLOAD_PRESETS:
                raise ValueError(f"Unknown workload '{w}'. Available: {list(WORKLOAD_PRESETS)}")
            if w == "unseen" and not self.cfg.allow_unseen:
                raise ValueError(
                    "The 'unseen' workload is reserved for generalization testing and is "
                    "blocked in this environment. Create an evaluation environment with "
                    "allow_unseen=True to use it."
                )
        for s in scenarios:
            if s not in NETWORK_SCENARIOS:
                raise ValueError(f"Unknown network scenario '{s}'. Available: {list(NETWORK_SCENARIOS)}")
            if s == "unseen" and not self.cfg.allow_unseen:
                raise ValueError(
                    "The 'unseen' network scenario is reserved for generalization testing and "
                    "is blocked in this environment. Use allow_unseen=True for evaluation."
                )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = dict(options or {})
        unknown = set(options) - _RESET_OPTION_KEYS
        if unknown:
            raise ValueError(
                f"Unknown reset option(s) {sorted(unknown)}. Valid: {sorted(_RESET_OPTION_KEYS)}"
            )

        # Always draw all three random values, so the RNG stream (and hence
        # reproducibility) does not depend on which options were supplied.
        drawn_workload = str(self.np_random.choice(self.cfg.workloads))
        drawn_scenario = str(self.np_random.choice(self.cfg.network_scenarios))
        drawn_task_seed = int(self.np_random.integers(0, 2**31 - 1))

        workload = options.get("workload_type", drawn_workload)
        scenario = options.get("network_scenario", drawn_scenario)
        duration_ms = float(options.get("duration_ms", self.cfg.duration_ms))
        task_seed = int(options.get("task_seed", drawn_task_seed))
        self._validate_choices([workload], [scenario])

        self.sim.reset(
            workload_type=workload,
            network_scenario=scenario,
            duration_ms=duration_ms,
            seed=task_seed,
        )
        self._reset_accumulators()
        self._ticks = 0
        self._energy_mark = 0.0  # servers were just reset; count energy from t=0
        self._needs_reset = False
        self._episode_meta = {
            "workload_type": workload,
            "network_scenario": scenario,
            "task_seed": task_seed,
            "duration_ms": duration_ms,
        }

        # Fast-forward to the first task that needs a decision. (If the tick
        # limit is hit here, the very first step() reports truncation.)
        self._truncated_at_reset = self._advance_to_next_decision(at_least_once=False)
        self._refresh_current_task()

        info = {
            **self._episode_meta,
            "num_tasks": len(self.sim.tasks),
            "action_mask": self._mask.copy(),
            "action_mask_level": self._mask_level,
        }
        return self._observation(), info

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("step() called before reset() or after the episode ended; call reset().")

        # Degenerate episode: no task ever needed a decision (empty task stream,
        # or the tick limit was hit while waiting for the first arrival).
        if self._task is None:
            self._needs_reset = True
            truncated = self._truncated_at_reset
            info = {
                "episode": self._episode_summary(truncated=truncated),
                "action_mask": self._mask.copy(),
                "action_mask_level": self._mask_level,
            }
            return self._observation(), 0.0, not truncated, truncated, info

        action = int(action)
        if not 0 <= action < len(self.server_ids):
            raise ValueError(f"Invalid action {action}; expected 0..{len(self.server_ids) - 1}.")

        sim = self.sim
        task = self._task
        action_was_valid = bool(self._mask[action])
        server_id = self.server_ids[action]

        n_completed_before = len(sim.completed_tasks)
        n_failed_before = len(sim.failed_tasks)

        # 1) apply the decision
        accepted = sim.submit_decision(task, server_id)

        # 2) load balance right after the decision (includes the new task in-flight)
        load_scores = self.state_builder.load_scores(
            sim, self.reward_config.load_cpu_weight, self.reward_config.load_queue_weight
        )

        # 3) advance time exactly as run_episode() does: once after the last
        #    pending decision of a tick, then until the next decision / end.
        truncated = False
        if not sim.has_pending_decision():
            truncated = self._advance_to_next_decision(at_least_once=True)

        # 4) collect what happened, compute reward
        energy_now = self._total_energy()
        energy_delta_j = max(0.0, energy_now - self._energy_mark)
        self._energy_mark = energy_now
        new_completed = sim.completed_tasks[n_completed_before:]
        new_failed = sim.failed_tasks[n_failed_before:]
        breakdown = self.reward_engine.compute(
            RewardInputs(
                completed=[
                    {
                        "latency_ms": c["latency_ms"],
                        "met_deadline": c["met_deadline"],
                        "priority": str(sim.task_lookup[c["task_id"]].priority),
                    }
                    for c in new_completed
                ],
                failed=[
                    {
                        "reason": f["reason"],
                        "priority": str(sim.task_lookup[f["task_id"]].priority),
                    }
                    for f in new_failed
                ],
                load_scores=load_scores,
                cpu_utilization=[sim.servers[sid].cpu_utilization() for sid in self.server_ids],
                cpu_cores=self._cpu_cores,
                energy_delta_j=energy_delta_j,
            )
        )

        # 5) bookkeeping
        self._num_decisions += 1
        self._episode_return += breakdown.total
        for term in TERMS:
            self._raw_sum[term] += breakdown.raw[term]
            self._weighted_sum[term] += breakdown.weighted[term]
        self._counts.update(breakdown.counts)
        self._failure_reasons.update(f["reason"] for f in new_failed)

        terminated = sim.is_episode_done()
        self._refresh_current_task()

        info = {
            "task_id": task.task_id,
            "server_id": server_id,
            "accepted": accepted,
            "action_was_valid": action_was_valid,
            "current_time_ms": sim.current_time_ms,
            "reward_raw": breakdown.raw,
            "reward_weighted": breakdown.weighted,
            **breakdown.counts,
            "action_mask": self._mask.copy(),
            "action_mask_level": self._mask_level,
        }
        if terminated or truncated:
            self._needs_reset = True
            info["episode"] = self._episode_summary(truncated=truncated)

        return self._observation(), float(breakdown.total), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Boolean mask for the CURRENT pending task (sb3-contrib MaskablePPO compatible)."""
        return self._mask.copy()

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Explainability helper
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        return list(self.state_builder.feature_names)

    def decode_observation(self, obs: np.ndarray) -> dict:
        return self.state_builder.decode(obs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _advance_to_next_decision(self, at_least_once: bool) -> bool:
        """
        Advance the simulator until a task needs a decision or the episode is
        done. If `at_least_once`, advances at least one tick first (this
        mirrors run_episode(), which always advances once after a tick's
        decisions). Returns True if the max_ticks safety limit was hit.
        """
        sim = self.sim
        first = at_least_once
        while first or (not sim.has_pending_decision() and not sim.is_episode_done()):
            first = False
            sim.advance_time()
            self._ticks += 1
            if self._ticks > self.cfg.max_ticks:
                return True
        return False

    def _refresh_current_task(self):
        self._task = self.sim.next_pending_task()
        self._mask, self._mask_level = self.state_builder.action_mask(self.sim, self._task)

    def _observation(self) -> np.ndarray:
        return self.state_builder.build(self.sim, self._task)

    def _total_energy(self) -> float:
        return sum(s.total_energy_joules for s in self.sim.servers.values())

    def _reset_accumulators(self):
        self._num_decisions = 0
        self._episode_return = 0.0
        self._raw_sum = {t: 0.0 for t in TERMS}
        self._weighted_sum = {t: 0.0 for t in TERMS}
        self._counts = Counter()
        self._failure_reasons = Counter()

    def _episode_summary(self, truncated: bool) -> dict:
        sim = self.sim
        summary = dict(sim.get_episode_summary())
        latencies = [c["latency_ms"] for c in sim.completed_tasks]
        late = sum(1 for c in sim.completed_tasks if not c["met_deadline"])
        deadline_failures = sum(
            1 for f in sim.failed_tasks if f["reason"] in DEADLINE_FAILURE_REASONS
        )
        summary.update(
            {
                **self._episode_meta,
                "truncated": truncated,
                "num_decisions": self._num_decisions,
                "episode_return": self._episode_return,
                "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
                "late_completions": late,
                "deadline_failures": deadline_failures,
                "sla_violations": late + deadline_failures,
                "rejections": summary["num_failed"] - deadline_failures,
                "failure_reasons": dict(self._failure_reasons),
                "reward_terms_raw": dict(self._raw_sum),
                "reward_terms_weighted": dict(self._weighted_sum),
            }
        )
        return summary


# ----------------------------------------------------------------------
# Baseline helper
# ----------------------------------------------------------------------

def rollout_with_scheduler(
    env: EdgeSchedulingEnv,
    scheduler,
    seed: int | None = None,
    options: dict | None = None,
) -> dict:
    """
    Run one full episode with a rule-based baseline (FIFO / RoundRobin /
    Random / Greedy - anything callable as scheduler(task, sim) -> server_id)
    through the Gymnasium environment, and return info["episode"].

    Because the baseline sees the SAME environment, task stream and reward as
    the DRL agent will, this gives directly comparable episode returns.
    """
    if hasattr(scheduler, "reset"):
        scheduler.reset()
    _, info = env.reset(seed=seed, options=options)
    while True:
        task = env.sim.next_pending_task()
        action = 0 if task is None else env.server_ids.index(scheduler(task, env.sim))
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info["episode"]