"""
episode_metrics.py

The evaluation metrics required by the research planner (Section 7) that the
simulator's own episode summary does not provide, computed from the simulator's
logs. Works for ANY scheduler (baseline or learned) because it only reads the
simulator state - it never looks at how decisions were made.

Planner metric                          -> key (all prefixed, default "m_")
--------------------------------------------------------------------------
Average latency                         -> latency_mean_ms
p95 / p99 latency                       -> latency_p95_ms, latency_p99_ms   (also p50, max)
Throughput                              -> throughput_tps, goodput_tps
Deadline / SLA violation rate           -> sla_violation_rate, on_time_rate
CPU / memory utilization                -> util_cpu_fleet, util_ram_fleet   (needs recorder)
Per-server utilization variance         -> util_variance_servers, util_std_servers (needs recorder)
Queue length / waiting time             -> queue_mean_fill, excess_latency_mean_ms
Task distribution / fairness            -> jain_server_tasks, jain_server_tasks_per_core,
                                           max_server_task_share, jain_user_completion
Energy proxy                            -> energy_per_task_j, mean_power_w
(plus per-server columns util_server<ID>, queue_server<ID>, tasks_server<ID>)

Two kinds of metric
-------------------
1. From the completion/failure logs alone (no setup needed): latency, SLA,
   throughput, fairness, energy.
2. Time-series metrics (utilization, queues): the simulator only knows its
   CURRENT utilization, so an `EpisodeRecorder` samples every server after every
   simulation tick. It wraps `sim.advance_time` on that ONE simulator instance,
   without editing the simulator's source.

Recorder usage (order matters):

    rec = EpisodeRecorder(env.sim)          # attach once
    ...
    rec.clear()                             # BEFORE env.reset(): reset fast-forwards
    env.reset(...)                          #   simulated time, and those ticks count
    ... run the episode ...
    metrics = compute_episode_metrics(env.sim, rec)

Definitions worth knowing
-------------------------
* Utilization and queue statistics cover the ARRIVAL WINDOW only (simulated time
  up to `duration_ms`). The drain tail afterwards is mostly idle and its length
  depends on the scheduler, so including it would flatter slow schedulers.
* Latency statistics are over COMPLETED tasks only; failed tasks have no latency
  and are captured by the SLA/rejection rates instead.
* `excess_latency` = latency - execution time = time lost to network transfer and
  queueing (the simulator does not record queue-wait separately).
* Jain's fairness index of x_1..x_n is (sum x)^2 / (n * sum x^2): 1.0 = perfectly
  even, 1/n = everything on one server. It is NaN when undefined (all zeros).
* Throughput divides by the total simulated time until the fleet drained
  (`makespan`), so a scheduler that leaves tasks stuck for a long time is
  penalised.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

DEADLINE_FAILURE = "deadline_missed_while_running"


# ----------------------------------------------------------------------
# small pure helpers
# ----------------------------------------------------------------------

def jain_index(values) -> float:
    """Jain's fairness index; NaN if there is nothing to compare (empty / all zero)."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan")
    total = x.sum()
    denom = x.size * np.sum(x * x)
    if total <= 0 or denom <= 0:
        return float("nan")
    return float(total * total / denom)


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


# ----------------------------------------------------------------------
# recorder
# ----------------------------------------------------------------------

class EpisodeRecorder:
    """Samples per-server CPU/RAM utilization and queue length after every tick."""

    def __init__(self, sim, attach: bool = True):
        self.sim = sim
        self._attached = False
        self.clear()
        if attach:
            self.attach()

    # -- lifecycle -------------------------------------------------------
    def attach(self) -> None:
        if self._attached:
            return
        if "advance_time" in self.sim.__dict__:
            raise RuntimeError("advance_time is already wrapped on this simulator "
                               "(another recorder attached?). Detach it first.")
        original = self.sim.advance_time   # bound method of the class

        def recorded_advance_time(delta_ms=None):
            result = original(delta_ms)
            delta = self.sim.time_step_ms if delta_ms is None else delta_ms
            self._sample(delta)
            return result

        self.sim.advance_time = recorded_advance_time
        self._attached = True

    def detach(self) -> None:
        if self._attached:
            self.sim.__dict__.pop("advance_time", None)   # falls back to the class method
            self._attached = False

    def clear(self) -> None:
        """Forget everything recorded so far (call before env.reset())."""
        self.times_ms: list[float] = []
        self.dt_ms: list[float] = []
        self.cpu_util: list[list[float]] = []
        self.ram_util: list[list[float]] = []
        self.queue_len: list[list[int]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()

    @property
    def num_ticks(self) -> int:
        return len(self.times_ms)

    # -- sampling ----------------------------------------------------------
    def _sample(self, delta_ms: float) -> None:
        ids = sorted(self.sim.servers)
        self.times_ms.append(float(self.sim.current_time_ms))
        self.dt_ms.append(float(delta_ms))
        self.cpu_util.append([self.sim.servers[i].cpu_utilization() for i in ids])
        self.ram_util.append([self.sim.servers[i].ram_utilization() for i in ids])
        self.queue_len.append([self.sim.servers[i].queue_length() for i in ids])


# ----------------------------------------------------------------------
# metric computation
# ----------------------------------------------------------------------

def compute_episode_metrics(sim, recorder: EpisodeRecorder | None = None, prefix: str = "m_") -> dict:
    """All metrics for the episode that just finished on `sim`, as {prefix+name: float}."""
    ids = sorted(sim.servers)
    servers = [sim.servers[i] for i in ids]
    cores = np.array([s.cpu_cores for s in servers], dtype=float)

    n_tasks = len(sim.tasks)
    completed, failed = sim.completed_tasks, sim.failed_tasks
    n_done, n_failed = len(completed), len(failed)
    denom = max(n_tasks, 1)
    out: dict[str, float] = {}

    # ---- latency (completed tasks) ----
    lat = np.array([c["latency_ms"] for c in completed], dtype=float)
    out["latency_mean_ms"] = float(lat.mean()) if lat.size else float("nan")
    out["latency_p50_ms"] = _percentile(lat, 50)
    out["latency_p95_ms"] = _percentile(lat, 95)
    out["latency_p99_ms"] = _percentile(lat, 99)
    out["latency_max_ms"] = float(lat.max()) if lat.size else float("nan")
    excess = np.array([c["latency_ms"] - sim.task_lookup[c["task_id"]].execution_time_ms
                       for c in completed], dtype=float)
    out["excess_latency_mean_ms"] = float(excess.mean()) if excess.size else float("nan")

    # ---- SLA / service rates (over ALL generated tasks) ----
    late = sum(1 for c in completed if not c["met_deadline"])
    deadline_failures = sum(1 for f in failed if f["reason"] == DEADLINE_FAILURE)
    rejections = n_failed - deadline_failures
    on_time = n_done - late
    out["on_time_rate"] = on_time / denom
    out["late_rate"] = late / denom
    out["deadline_failure_rate"] = deadline_failures / denom
    out["rejection_rate"] = rejections / denom
    out["sla_violation_rate"] = (late + deadline_failures) / denom
    out["completion_rate"] = n_done / denom

    # ---- throughput ----
    makespan_s = sim.current_time_ms / 1000.0
    out["makespan_s"] = float(makespan_s)
    out["throughput_tps"] = n_done / makespan_s if makespan_s > 0 else float("nan")
    out["goodput_tps"] = on_time / makespan_s if makespan_s > 0 else float("nan")

    # ---- energy ----
    energy = float(sum(s.total_energy_joules for s in servers))
    out["energy_j"] = energy
    out["energy_per_task_j"] = energy / denom
    out["mean_power_w"] = energy / makespan_s if makespan_s > 0 else float("nan")

    # ---- task distribution / fairness ----
    counts = Counter(c["server_id"] for c in completed)
    per_server_tasks = np.array([counts.get(i, 0) for i in ids], dtype=float)
    out["jain_server_tasks"] = jain_index(per_server_tasks)
    out["jain_server_tasks_per_core"] = jain_index(per_server_tasks / cores)
    out["max_server_task_share"] = (float(per_server_tasks.max() / per_server_tasks.sum())
                                    if per_server_tasks.sum() > 0 else float("nan"))
    for sid, n in zip(ids, per_server_tasks):
        out[f"tasks_server{sid}"] = float(n)

    ratios = [st["tasks_completed"] / st["tasks_submitted"]
              for st in sim.user_stats.values() if st["tasks_submitted"] > 0]
    out["jain_user_completion"] = jain_index(ratios) if ratios else float("nan")
    out["min_user_completion_rate"] = float(min(ratios)) if ratios else float("nan")

    # ---- utilization and queues (need the recorder), arrival window only ----
    nan = float("nan")
    util_keys = ("util_cpu_fleet", "util_ram_fleet", "util_variance_servers", "util_std_servers",
                 "util_std_instant_mean", "queue_mean_fill", "queue_max_seen")
    have = False
    if recorder is not None and recorder.num_ticks > 0:
        t = np.array(recorder.times_ms)
        window = t <= sim.duration_ms
        if window.any():
            have = True
            w = np.array(recorder.dt_ms)[window]
            cpu = np.array(recorder.cpu_util)[window]
            ram = np.array(recorder.ram_util)[window]
            queue = np.array(recorder.queue_len, dtype=float)[window]
            max_q = np.array([s.max_queue_length for s in servers], dtype=float)
            ram_gb = np.array([s.ram_gb for s in servers], dtype=float)

            cpu_mean = np.average(cpu, axis=0, weights=w)
            ram_mean = np.average(ram, axis=0, weights=w)
            queue_mean = np.average(queue, axis=0, weights=w)
            out["util_cpu_fleet"] = float(np.dot(cpu_mean, cores) / cores.sum())
            out["util_ram_fleet"] = float(np.dot(ram_mean, ram_gb) / ram_gb.sum())
            out["util_variance_servers"] = float(np.var(cpu_mean))
            out["util_std_servers"] = float(np.std(cpu_mean))
            out["util_std_instant_mean"] = float(np.average(np.std(cpu, axis=1), weights=w))
            out["queue_mean_fill"] = float(np.mean(queue_mean / max_q))
            out["queue_max_seen"] = float(queue.max())
            for sid, u, q in zip(ids, cpu_mean, queue_mean):
                out[f"util_server{sid}"] = float(u)
                out[f"queue_server{sid}"] = float(q)
    if not have:
        for k in util_keys:
            out[k] = nan
        for sid in ids:
            out[f"util_server{sid}"] = nan
            out[f"queue_server{sid}"] = nan

    return {f"{prefix}{k}": float(v) for k, v in out.items()}