"""
state_builder.py

Turns the simulator's raw state into what the DRL agent sees:

  1. build()        -> a fixed-size, normalised float32 observation vector
  2. action_mask()  -> which servers are legal / sensible for the pending task
  3. feature_names  -> a name for every observation index (for explainability)
  4. decode()       -> a readable {task: ..., servers: [...]} view of a vector

Observation layout (all values normalised; see env_config.yaml for refs)
------------------------------------------------------------------------
Task block (9 values) - the task waiting for a decision:
    task_cpu, task_ram          requirement / fleet-max capacity, in [0, 1]
    task_data_size              MB / data_size_ref_mb, clipped to [0, 1]
    task_exec_time              ms / exec_time_ref_ms, clipped to [0, 1]
    task_slack                  (deadline - now) / exec_time / slack_ref,
                                clipped to [-1, 1]. Negative = already late.
    task_prio_high/medium/low   one-hot priority
    pending_decisions           waiting decisions / pending_ref, clipped [0, 1]

Then one block per server, in ascending server_id order (10 values, or 11
with include_feasibility):
    cpu_util, ram_util          current utilization, [0, 1]
    queue_fill                  (queued + in-flight) / max_queue, clipped [0, 1]
    backlog                     estimated ms to drain all work already
                                committed to this server, squashed to [0, 1)
    net_delay                   EXACT network delay this task would incur on
                                this server right now, squashed to [0, 1)
    can_start_now               1 if the task would start immediately (free
                                CPU/RAM and nothing ahead of it), else 0
    feasible                    (optional) the action-mask bit
    cpu_cores, ram_gb,          static capacity / fleet max
    bandwidth, power_max        (lets one policy handle heterogeneity)

Why net_delay and backlog: in this simulator a task's deadline budget does
not include network transfer time, and transfer time grows with the number of
transfers already in flight to a server. Delay and backlog are therefore the
two quantities that most directly decide whether a deadline is met.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, fields

import numpy as np

PRIORITIES: tuple[str, ...] = ("high", "medium", "low")

TASK_FEATURES: tuple[str, ...] = (
    "task_cpu",
    "task_ram",
    "task_data_size",
    "task_exec_time",
    "task_slack",
    "task_prio_high",
    "task_prio_medium",
    "task_prio_low",
    "pending_decisions",
)


# ----------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------

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
class ObservationConfig:
    data_size_ref_mb: float = 50.0
    exec_time_ref_ms: float = 2000.0
    slack_ref: float = 5.0
    delay_ref_ms: float = 500.0
    backlog_ref_ms: float = 1000.0
    pending_ref: float = 10.0
    include_feasibility: bool = False

    def __post_init__(self):
        for name in (
            "data_size_ref_mb", "exec_time_ref_ms", "slack_ref",
            "delay_ref_ms", "backlog_ref_ms", "pending_ref",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"observation.{name} must be > 0.")

    @classmethod
    def from_dict(cls, data) -> "ObservationConfig":
        return _build_dataclass(cls, data, "observation config")


@dataclass(frozen=True)
class MaskConfig:
    overload_fraction: float = 1.0

    def __post_init__(self):
        if not (0.0 < self.overload_fraction <= 1.0):
            raise ValueError("action_mask.overload_fraction must be in (0, 1].")

    @classmethod
    def from_dict(cls, data) -> "MaskConfig":
        return _build_dataclass(cls, data, "action_mask config")


def _squash(x: float, ref: float) -> float:
    """Map [0, inf) -> [0, 1) monotonically; non-finite values map to 1."""
    if not math.isfinite(x) or x < 0:
        return 1.0 if not math.isfinite(x) else 0.0
    return x / (x + ref)


# ----------------------------------------------------------------------
# State builder
# ----------------------------------------------------------------------

class StateBuilder:
    """
    Reads (never modifies) an EdgeEnvironment and produces observations and
    action masks. Constructed once per environment: it caches fleet-wide
    maxima so per-step work is small.
    """

    def __init__(self, sim, obs_config: ObservationConfig | None = None,
                 mask_config: MaskConfig | None = None):
        self.obs_config = obs_config or ObservationConfig()
        self.mask_config = mask_config or MaskConfig()

        self.server_ids: list[int] = sorted(sim.servers)
        self.num_servers = len(self.server_ids)

        servers = [sim.servers[sid] for sid in self.server_ids]
        self._max_cpu = max(s.cpu_cores for s in servers)
        self._max_ram = max(s.ram_gb for s in servers)
        self._max_bw = max(s.bandwidth_mbps for s in servers)
        self._max_power = max(s.power_max_w for s in servers)

        per_server = [
            "cpu_util", "ram_util", "queue_fill", "backlog", "net_delay", "can_start_now",
        ]
        if self.obs_config.include_feasibility:
            per_server.append("feasible")
        per_server += ["cpu_cores", "ram_gb", "bandwidth", "power_max"]
        self.server_features: tuple[str, ...] = tuple(per_server)

        self.obs_dim = len(TASK_FEATURES) + self.num_servers * len(self.server_features)
        self.feature_names: list[str] = list(TASK_FEATURES) + [
            f"server{sid}_{feat}" for sid in self.server_ids for feat in self.server_features
        ]

        # Bounds: everything is in [0, 1] except task_slack in [-1, 1].
        low = np.zeros(self.obs_dim, dtype=np.float32)
        high = np.ones(self.obs_dim, dtype=np.float32)
        low[TASK_FEATURES.index("task_slack")] = -1.0
        self.low, self.high = low, high

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def build(self, sim, task) -> np.ndarray:
        """Observation for `task` (the task awaiting a decision). If `task` is
        None (no decision pending), returns an all-zero vector."""
        if task is None:
            return np.zeros(self.obs_dim, dtype=np.float32)

        cfg = self.obs_config
        obs = np.zeros(self.obs_dim, dtype=np.float64)

        # ---- task block ----
        slack_factor = (task.deadline_ms - sim.current_time_ms) / max(task.execution_time_ms, 1e-9)
        obs[0] = task.cpu_requirement / self._max_cpu
        obs[1] = task.ram_requirement / self._max_ram
        obs[2] = task.data_size_mb / cfg.data_size_ref_mb
        obs[3] = task.execution_time_ms / cfg.exec_time_ref_ms
        obs[4] = slack_factor / cfg.slack_ref
        for i, prio in enumerate(PRIORITIES):
            obs[5 + i] = 1.0 if str(task.priority) == prio else 0.0
        obs[8] = len(sim.pending_queue) / cfg.pending_ref

        # ---- per-server blocks ----
        states = sim.get_server_states()
        inflight_work = defaultdict(float)  # core*ms of work still in network transit
        for entry in sim.pending_transfers:
            t = entry["task"]
            inflight_work[entry["server_id"]] += t.execution_time_ms * t.cpu_requirement

        idx = len(TASK_FEATURES)
        for sid in self.server_ids:
            srv = sim.servers[sid]
            st = states[sid]

            queue_fill = (
                st["effective_queue_length"] / srv.max_queue_length
                if srv.max_queue_length > 0 else 1.0
            )
            work = (
                sum(t.remaining_time_ms * t.cpu_requirement for t in srv.running_tasks)
                + sum(t.remaining_time_ms * t.cpu_requirement for t in srv.queue)
                + inflight_work.get(sid, 0.0)
            )
            backlog_ms = work / srv.cpu_cores if srv.cpu_cores > 0 else float("inf")
            can_start_now = (
                task.cpu_requirement <= srv.available_cpu()
                and task.ram_requirement <= srv.available_ram()
                and st["queue_length"] == 0
                and st["in_flight_tasks"] == 0
            )

            values = [
                st["cpu_utilization"],
                st["ram_utilization"],
                queue_fill,
                _squash(backlog_ms, cfg.backlog_ref_ms),
                _squash(self.preview_network_delay_ms(sim, task, srv), cfg.delay_ref_ms),
                1.0 if can_start_now else 0.0,
            ]
            if cfg.include_feasibility:
                values.append(1.0 if self._fits(task, srv) and queue_fill < 1.0 else 0.0)
            values += [
                srv.cpu_cores / self._max_cpu,
                srv.ram_gb / self._max_ram,
                srv.bandwidth_mbps / self._max_bw,
                srv.power_max_w / self._max_power,
            ]
            obs[idx: idx + len(values)] = values
            idx += len(values)

        return np.clip(obs, self.low, self.high).astype(np.float32)

    @staticmethod
    def preview_network_delay_ms(sim, task, server) -> float:
        """
        The exact network delay `task` would incur if submitted to `server`
        right now: latency + transfer time, INCLUDING the effect of this task's
        own transfer (submit_decision calls start_transfer before timing it).
        Implemented by starting and immediately ending a transfer on the real
        Network object, so it can never drift from the simulator's formula.
        The Network is restored to its previous transfer count.
        """
        net = sim.network
        net.start_transfer(server.server_id)
        try:
            return net.total_network_delay_ms(task.data_size_mb, server)
        finally:
            net.end_transfer(server.server_id)

    # ------------------------------------------------------------------
    # Action mask
    # ------------------------------------------------------------------

    @staticmethod
    def _fits(task, server) -> bool:
        """Can this task EVER run on this server (static capacity check)?"""
        return (
            task.cpu_requirement <= server.cpu_cores
            and task.ram_requirement <= server.ram_gb
        )

    def action_mask(self, sim, task) -> tuple[np.ndarray, int]:
        """
        Returns (mask, level). mask[i] is True if action i (server_ids[i]) is
        allowed. `level` says how strict the mask had to be:

          0  fits AND effective queue < overload_fraction * max_queue  (normal)
          1  fits AND effective queue < max_queue   (overload pruning relaxed)
          2  fits only                              (every server is full; the
                                                    simulator will reject/fail)
          3  everything allowed                     (task fits nowhere)

        The mask is never empty. Levels 2-3 exist so the agent always has a
        legal action; the resulting failure is then penalised by the reward.
        Effective queue = queued + in-flight, so the mask is stricter than the
        simulator's own can_accept() (which ignores in-flight tasks) and
        prevents "queue_full_on_arrival" failures by construction.
        """
        n = self.num_servers
        if task is None:
            return np.ones(n, dtype=bool), 3

        states = sim.get_server_states()
        fits = np.zeros(n, dtype=bool)
        eff_q = np.zeros(n)
        max_q = np.zeros(n)
        for i, sid in enumerate(self.server_ids):
            srv = sim.servers[sid]
            fits[i] = self._fits(task, srv)
            eff_q[i] = states[sid]["effective_queue_length"]
            max_q[i] = srv.max_queue_length

        frac = self.mask_config.overload_fraction
        candidates = (
            (0, fits & (eff_q < max_q * frac)),
            (1, fits & (eff_q < max_q)),
            (2, fits),
        )
        for level, mask in candidates:
            if mask.any():
                return mask, level
        return np.ones(n, dtype=bool), 3

    # ------------------------------------------------------------------
    # Load scores (used by the balance reward term)
    # ------------------------------------------------------------------

    def load_scores(self, sim, cpu_weight: float = 0.5, queue_weight: float = 0.5) -> list[float]:
        """Per-server load = cpu_w*cpu_util + queue_w*(effective queue / max queue)."""
        states = sim.get_server_states()
        scores = []
        for sid in self.server_ids:
            srv = sim.servers[sid]
            st = states[sid]
            qf = (
                st["effective_queue_length"] / srv.max_queue_length
                if srv.max_queue_length > 0 else 1.0
            )
            scores.append(cpu_weight * st["cpu_utilization"] + queue_weight * qf)
        return scores

    # ------------------------------------------------------------------
    # Explainability helper
    # ------------------------------------------------------------------

    def decode(self, obs: np.ndarray) -> dict:
        """Readable view of an observation vector: {'task': {...}, 'servers': {id: {...}}}."""
        obs = np.asarray(obs, dtype=float)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"Expected shape ({self.obs_dim},), got {obs.shape}.")
        task = {name: float(obs[i]) for i, name in enumerate(TASK_FEATURES)}
        servers = {}
        k = len(self.server_features)
        for j, sid in enumerate(self.server_ids):
            start = len(TASK_FEATURES) + j * k
            servers[sid] = {
                feat: float(obs[start + m]) for m, feat in enumerate(self.server_features)
            }
        return {"task": task, "servers": servers}