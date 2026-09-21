"""
test_metrics.py

Tests for metrics/episode_metrics.py. Simulator-only: no PyTorch, Gymnasium or
Stable-Baselines needed.

Run from the project root:
    python test_metrics.py
    python -m pytest test_metrics.py -v
"""

import math
import sys
import traceback
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from environment.edge_environment import EdgeEnvironment
from environment.task_generator import Task
from metrics.episode_metrics import EpisodeRecorder, compute_episode_metrics, jain_index
from scheduling.ect_scheduler import EarliestCompletionScheduler
from scheduling.fifo import FIFOScheduler
from scheduling.greedy_scheduler import GreedyScheduler
from scheduling.round_robin import RoundRobinScheduler

ROOT = Path(__file__).resolve().parent
SERVER_YAML = ROOT / "config" / "server_config.yaml"


@contextmanager
def expect_raises(exc_type, contains=None):
    try:
        yield
    except exc_type as e:
        if contains is not None:
            assert contains in str(e), f"expected '{contains}' in error message, got: {e}"
    else:
        raise AssertionError(f"expected {exc_type.__name__} to be raised")


def server_cfgs():
    with open(SERVER_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["servers"]


def new_sim():
    return EdgeEnvironment(server_cfgs(), num_users=15, time_step_ms=10.0)


def run_with_recorder(scheduler, workload="normal", network="normal", seed=3, duration=10000.0):
    sim = new_sim()
    rec = EpisodeRecorder(sim)
    rec.clear()
    sim.run_episode(scheduler, workload, network, duration, seed=seed)
    return sim, rec, compute_episode_metrics(sim, rec)


def fake_sim(completed, failed, tasks, n_servers=2, current_time_ms=10000.0, duration_ms=8000.0):
    """A minimal stand-in exposing exactly what compute_episode_metrics reads."""
    servers = {i + 1: SimpleNamespace(cpu_cores=4.0, ram_gb=8.0, max_queue_length=10,
                                      total_energy_joules=100.0) for i in range(n_servers)}
    return SimpleNamespace(
        servers=servers, tasks=tasks, completed_tasks=completed, failed_tasks=failed,
        task_lookup={t.task_id: t for t in tasks}, user_stats={},
        current_time_ms=current_time_ms, duration_ms=duration_ms)


def mk_task(i, exec_ms=100.0):
    return Task(task_id=i, user_id=0, cpu_requirement=1.0, ram_requirement=1.0, data_size_mb=1.0,
                execution_time_ms=exec_ms, deadline_ms=1e9, priority="medium", arrival_time_ms=0.0)


def done(i, server_id, latency, met=True):
    return {"task_id": i, "user_id": 0, "server_id": server_id, "latency_ms": latency, "met_deadline": met}


# ----------------------------------------------------------------------
# pure functions
# ----------------------------------------------------------------------

def test_jain_index_properties():
    assert math.isclose(jain_index([5, 5, 5, 5]), 1.0)
    assert math.isclose(jain_index([12, 0, 0, 0]), 0.25)          # everything on one of 4 -> 1/n
    assert 0.25 < jain_index([8, 4, 2, 1]) < 1.0
    assert math.isclose(jain_index([3, 3]), jain_index([30, 30]))  # scale invariant
    assert math.isnan(jain_index([]))
    assert math.isnan(jain_index([0, 0, 0]))                       # undefined, not "perfectly fair"


def test_latency_sla_and_throughput_from_hand_built_logs():
    tasks = [mk_task(i, exec_ms=100.0) for i in range(1, 11)]                # 10 tasks generated
    completed = [done(i, 1 if i % 2 else 2, latency=100.0 * i, met=(i <= 6)) for i in range(1, 9)]  # 8 done
    failed = [{"task_id": 9, "user_id": 0, "reason": "deadline_missed_while_running", "time_ms": 1.0},
              {"task_id": 10, "user_id": 0, "reason": "server_full_or_incompatible", "time_ms": 1.0}]
    m = compute_episode_metrics(fake_sim(completed, failed, tasks), prefix="")

    lat = np.array([100.0 * i for i in range(1, 9)])
    assert math.isclose(m["latency_mean_ms"], lat.mean())
    assert math.isclose(m["latency_p95_ms"], np.percentile(lat, 95))
    assert math.isclose(m["latency_p99_ms"], np.percentile(lat, 99))
    assert math.isclose(m["latency_max_ms"], 800.0)
    assert math.isclose(m["excess_latency_mean_ms"], lat.mean() - 100.0)
    # 8 completed: 6 on time + 2 late; 1 deadline failure; 1 rejection; out of 10 generated
    assert math.isclose(m["on_time_rate"], 0.6) and math.isclose(m["late_rate"], 0.2)
    assert math.isclose(m["deadline_failure_rate"], 0.1) and math.isclose(m["rejection_rate"], 0.1)
    assert math.isclose(m["sla_violation_rate"], 0.3)
    assert math.isclose(m["on_time_rate"] + m["sla_violation_rate"] + m["rejection_rate"], 1.0)
    assert math.isclose(m["completion_rate"], 0.8)
    # 8 completed in 10 s of simulated time; 6 of them on time
    assert math.isclose(m["throughput_tps"], 0.8) and math.isclose(m["goodput_tps"], 0.6)
    # 4 tasks per server -> perfectly even
    assert math.isclose(m["jain_server_tasks"], 1.0)
    assert math.isclose(m["max_server_task_share"], 0.5)
    assert m["tasks_server1"] == 4 and m["tasks_server2"] == 4
    # energy: 2 servers x 100 J over 10 tasks
    assert math.isclose(m["energy_j"], 200.0) and math.isclose(m["energy_per_task_j"], 20.0)
    assert math.isclose(m["mean_power_w"], 20.0)


def test_an_episode_with_no_completed_tasks_does_not_crash():
    tasks = [mk_task(1), mk_task(2)]
    failed = [{"task_id": 1, "user_id": 0, "reason": "server_full_or_incompatible", "time_ms": 0.0},
              {"task_id": 2, "user_id": 0, "reason": "server_full_or_incompatible", "time_ms": 0.0}]
    m = compute_episode_metrics(fake_sim([], failed, tasks), prefix="")
    assert math.isnan(m["latency_mean_ms"]) and math.isnan(m["latency_p95_ms"])
    assert m["on_time_rate"] == 0.0 and m["rejection_rate"] == 1.0
    assert math.isnan(m["jain_server_tasks"]) and math.isnan(m["max_server_task_share"])
    assert math.isnan(m["util_cpu_fleet"])       # no recorder -> undefined, not zero
    assert all(isinstance(v, float) for v in m.values())


# ----------------------------------------------------------------------
# recorder
# ----------------------------------------------------------------------

def test_recorder_attaches_samples_every_tick_and_detaches_cleanly():
    sim = new_sim()
    rec = EpisodeRecorder(sim)
    sim.reset("normal", "normal", 1000.0, seed=1)
    for _ in range(7):
        sim.advance_time()
    assert rec.num_ticks == 7
    assert rec.times_ms == [10.0 * (i + 1) for i in range(7)]
    assert all(len(row) == 4 for row in rec.cpu_util)

    with expect_raises(RuntimeError, "already wrapped"):
        EpisodeRecorder(sim)                      # a second recorder must not stack silently
    rec.detach()
    assert "advance_time" not in sim.__dict__      # original method restored
    sim.advance_time()
    assert rec.num_ticks == 7                     # no longer recording
    rec.attach()
    sim.advance_time()
    assert rec.num_ticks == 8
    rec.clear()
    assert rec.num_ticks == 0


def test_recorded_utilization_reproduces_the_simulators_energy_exactly():
    """The strongest check that the recorder samples the right thing: energy is
    sum over ticks and servers of P(utilization) * dt, so recorded per-tick
    utilizations must add up to the simulator's own energy counter."""
    for sched in (GreedyScheduler(), EarliestCompletionScheduler(), FIFOScheduler()):
        sim, rec, _ = run_with_recorder(sched, "heavy", "congested", seed=5, duration=4000.0)
        ids = sorted(sim.servers)
        dt_s = np.array(rec.dt_ms) / 1000.0
        cpu = np.array(rec.cpu_util)
        energy = 0.0
        for j, sid in enumerate(ids):
            s = sim.servers[sid]
            power = s.power_idle_w + (s.power_max_w - s.power_idle_w) * cpu[:, j] ** s.power_exponent
            energy += float(np.sum(power * dt_s))
        assert math.isclose(energy, sum(s.total_energy_joules for s in sim.servers.values()),
                            rel_tol=1e-9), type(sched).__name__


def test_recorder_covers_every_tick_of_a_full_episode():
    sim, rec, _ = run_with_recorder(GreedyScheduler(), "normal", "normal", seed=2, duration=3000.0)
    assert math.isclose(rec.times_ms[-1], sim.current_time_ms)
    assert rec.num_ticks == round(sim.current_time_ms / 10.0)      # none missed, none doubled


# ----------------------------------------------------------------------
# real episodes
# ----------------------------------------------------------------------

def test_metrics_on_real_episodes_are_consistent_and_in_range():
    for sched in (GreedyScheduler(), EarliestCompletionScheduler(), RoundRobinScheduler()):
        sim, rec, m = run_with_recorder(sched, "normal", "normal", seed=11)
        assert m["m_on_time_rate"] + m["m_sla_violation_rate"] + m["m_rejection_rate"] > 0.999
        assert 0.0 <= m["m_util_cpu_fleet"] <= 1.0 and 0.0 <= m["m_util_ram_fleet"] <= 1.0
        assert m["m_util_variance_servers"] >= 0.0
        assert math.isclose(m["m_util_std_servers"] ** 2, m["m_util_variance_servers"], rel_tol=1e-9, abs_tol=1e-15)
        assert 0.0 <= m["m_queue_mean_fill"] <= 1.5
        assert m["m_latency_p50_ms"] <= m["m_latency_p95_ms"] <= m["m_latency_p99_ms"] <= m["m_latency_max_ms"]
        assert 0.25 - 1e-9 <= m["m_jain_server_tasks"] <= 1.0
        assert math.isclose(sum(m[f"m_tasks_server{i}"] for i in (1, 2, 3, 4)), len(sim.completed_tasks))
        assert m["m_throughput_tps"] >= m["m_goodput_tps"] >= 0.0
        # the window excludes the drain tail, so recorded ticks exceed the window's ticks
        assert rec.num_ticks >= int(sim.duration_ms / 10.0) - 1
        assert all(isinstance(v, float) for v in m.values())


def test_fairness_metric_separates_fifo_from_round_robin():
    """FIFO funnels work to the first server; Round Robin spreads it by count."""
    _, _, fifo = run_with_recorder(FIFOScheduler(), "normal", "normal", seed=7)
    _, _, rr = run_with_recorder(RoundRobinScheduler(), "normal", "normal", seed=7)
    assert rr["m_jain_server_tasks"] > fifo["m_jain_server_tasks"] + 0.2
    assert fifo["m_max_server_task_share"] > rr["m_max_server_task_share"]


def test_prefix_and_repeatability():
    _, _, a = run_with_recorder(GreedyScheduler(), "variable", "congested", seed=4)
    _, _, b = run_with_recorder(GreedyScheduler(), "variable", "congested", seed=4)
    assert a == b or all((math.isnan(a[k]) and math.isnan(b[k])) or a[k] == b[k] for k in a)
    assert all(k.startswith("m_") for k in a)
    sim, rec, _ = run_with_recorder(GreedyScheduler(), "normal", "normal", seed=4)
    assert all(not k.startswith("m_") for k in compute_episode_metrics(sim, rec, prefix=""))


# ----------------------------------------------------------------------
# tiny runner so this file also works as `python test_metrics.py`
# ----------------------------------------------------------------------

def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except unittest.SkipTest as e:
            print(f"SKIP  {name}: {e}")
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed" + ("" if not failures else f", {failures} FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())