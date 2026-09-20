"""
test_ect_scheduler.py

Tests for scheduling/ect_scheduler.py (Earliest-Completion-Time baseline).
Uses only the simulator: no PyTorch or Gymnasium needed.

Run from the project root:
    python test_ect_scheduler.py
    python -m pytest test_ect_scheduler.py -v
"""

import sys
import traceback
import unittest
from pathlib import Path

import numpy as np
import yaml

from environment.edge_environment import EdgeEnvironment
from environment.task_generator import Task
from scheduling.ect_scheduler import EarliestCompletionScheduler
from scheduling.fifo import FIFOScheduler
from scheduling.greedy_scheduler import GreedyScheduler

ROOT = Path(__file__).resolve().parent
SERVER_YAML = ROOT / "config" / "server_config.yaml"


def server_cfgs():
    with open(SERVER_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["servers"]


def fresh_sim(cfgs=None, network="normal", seed=0):
    sim = EdgeEnvironment(cfgs or server_cfgs(), num_users=15, time_step_ms=10.0)
    sim.reset("normal", network, 10000.0, seed=seed)
    sim.pending_queue = []  # we place tasks by hand
    return sim


def make_task(task_id=1, cpu=0.5, ram=0.5, data_mb=1.0, exec_ms=300.0, deadline_ms=1e9):
    return Task(task_id=task_id, user_id=0, cpu_requirement=cpu, ram_requirement=ram,
                data_size_mb=data_mb, execution_time_ms=exec_ms, deadline_ms=deadline_ms,
                priority="medium", arrival_time_ms=0.0)


def load_server(sim, server_id, n_tasks, cpu=2.0, exec_ms=800.0, id_base=1000):
    """Put n_tasks in flight to a server (as a scheduler's decisions would)."""
    for i in range(n_tasks):
        t = make_task(id_base + i, cpu=cpu, ram=0.5, data_mb=1.0, exec_ms=exec_ms)
        sim.pending_queue = [t]
        sim.submit_decision(t, server_id)
    sim.pending_queue = []


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------

def test_prefers_the_fast_link_on_an_idle_fleet_where_greedy_does_not():
    sim = fresh_sim()
    task = make_task(data_mb=10.0, exec_ms=300.0)
    # Idle fleet: only the network differs. Server 4 has the fastest link for
    # big data (~287 ms vs 807 / 414 / 1604 ms on servers 1 / 2 / 3).
    assert EarliestCompletionScheduler()(task, sim) == 4
    # Greedy sees four equally idle servers and takes the first one.
    assert GreedyScheduler()(task, sim) == 1
    assert FIFOScheduler()(task, sim) == 1


def test_estimate_on_idle_server_is_network_delay_plus_execution():
    sim = fresh_sim()
    task = make_task(data_mb=6.0, exec_ms=400.0)
    sched = EarliestCompletionScheduler()
    for sid, server in sim.servers.items():
        sim.network.start_transfer(sid)
        delay = sim.network.total_network_delay_ms(task.data_size_mb, server)
        sim.network.end_transfer(sid)
        assert np.isclose(sched.estimate_completion_ms(task, sim, sid), delay + 400.0), sid


def test_avoids_a_server_with_a_large_backlog():
    sim = fresh_sim()
    load_server(sim, 4, n_tasks=10)             # 10 x (2 cores x 800 ms) queued up for server 4
    task = make_task(data_mb=0.5, exec_ms=200.0)
    sched = EarliestCompletionScheduler()
    assert sched(task, sim) != 4
    assert sched.estimate_completion_ms(task, sim, 4) > sched.estimate_completion_ms(task, sim, 3)


def test_skips_servers_the_task_cannot_fit_on_or_that_are_full():
    sim = fresh_sim()
    sched = EarliestCompletionScheduler()

    # Server 3 (2 cores) has the lowest latency but cannot host a 3-core task.
    big = make_task(cpu=3.0, data_mb=0.2, exec_ms=100.0)
    assert sched(big, sim) != 3

    # Fill server 3's queue (limit 15) with in-flight tasks: it becomes ineligible.
    sim2 = fresh_sim()
    max_q = sim2.servers[3].max_queue_length
    load_server(sim2, 3, n_tasks=max_q, cpu=0.1, exec_ms=50.0)
    assert sim2.get_server_states()[3]["effective_queue_length"] == max_q
    assert sched(make_task(data_mb=0.1, exec_ms=50.0, cpu=0.1), sim2) != 3


def test_estimating_does_not_disturb_network_state():
    sim = fresh_sim()
    load_server(sim, 2, n_tasks=3)
    before = {sid: sim.network.get_active_transfers(sid) for sid in sim.servers}
    EarliestCompletionScheduler()(make_task(data_mb=8.0), sim)
    after = {sid: sim.network.get_active_transfers(sid) for sid in sim.servers}
    assert before == after


def test_fallback_when_no_server_is_eligible():
    tiny = [dict(c, max_queue_length=1) for c in server_cfgs()[:2]]
    sim = fresh_sim(tiny)
    load_server(sim, 1, n_tasks=1, cpu=0.1, exec_ms=50.0, id_base=10)
    load_server(sim, 2, n_tasks=1, cpu=0.1, exec_ms=50.0, id_base=20)
    choice = EarliestCompletionScheduler()(make_task(), sim)
    assert choice in sim.servers  # a valid id, so the simulator can record a clean failure

    # ...and whole episodes on an overloaded tiny fleet still run to the end.
    env = EdgeEnvironment(tiny, num_users=15, time_step_ms=10.0)
    summary = env.run_episode(EarliestCompletionScheduler(), "heavy", "congested", 5000.0, seed=3)
    assert summary["total_tasks_generated"] == summary["num_completed"] + summary["num_failed"]


def test_is_deterministic_and_stateless():
    sim = fresh_sim()
    load_server(sim, 1, n_tasks=4)
    task = make_task(data_mb=3.0)
    sched = EarliestCompletionScheduler()
    assert len({sched(task, sim) for _ in range(5)}) == 1

    def run():
        env = EdgeEnvironment(server_cfgs(), num_users=15, time_step_ms=10.0)
        return env.run_episode(EarliestCompletionScheduler(), "burst", "normal", 10000.0, seed=9)

    assert run() == run()


def test_full_episodes_complete_on_every_workload_and_network():
    for workload in ("normal", "heavy", "burst", "variable"):
        for network in ("normal", "congested"):
            env = EdgeEnvironment(server_cfgs(), num_users=15, time_step_ms=10.0)
            s = env.run_episode(EarliestCompletionScheduler(), workload, network, 10000.0, seed=4)
            assert s["total_tasks_generated"] == s["num_completed"] + s["num_failed"], (workload, network)


def _on_time_rate(scheduler_factory, workload, seeds):
    rates = []
    for seed in seeds:
        env = EdgeEnvironment(server_cfgs(), num_users=15, time_step_ms=10.0)
        env.run_episode(scheduler_factory(), workload, "normal", 10000.0, seed=seed)
        rates.append(sum(1 for c in env.completed_tasks if c["met_deadline"]) / max(len(env.tasks), 1))
    return float(np.mean(rates))


def test_beats_fifo_and_greedy_on_normal_load():
    """The property this baseline exists for (measured: about 0.85 vs 0.65 on-time
    for Greedy under normal load). Margins here are deliberately conservative."""
    seeds = range(700_000_000, 700_000_008)
    ect = _on_time_rate(EarliestCompletionScheduler, "normal", seeds)
    greedy = _on_time_rate(GreedyScheduler, "normal", seeds)
    fifo = _on_time_rate(FIFOScheduler, "normal", seeds)
    assert ect > fifo + 0.3, (ect, fifo)
    assert ect > greedy + 0.05, (ect, greedy)


# ----------------------------------------------------------------------
# tiny runner so this file also works as `python test_ect_scheduler.py`
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