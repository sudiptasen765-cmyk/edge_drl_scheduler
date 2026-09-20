"""
test_drl_env.py

Tests for environment/state_builder.py and environment/edge_scheduling_env.py.

Run either way, from the project root:
    python test_drl_env.py
    python -m pytest test_drl_env.py -v

The most important tests:
  * test_baselines_match_run_episode   - the Gym wrapper does not change the
    simulator's behaviour (same decisions -> identical results).
  * test_task_conservation             - every task ends up completed or
    failed exactly once, and no reward event is counted twice or dropped.
  * test_masked_policy_is_never_rejected - the action mask really prevents
    rejections.
  * test_gymnasium_check_env           - the official Gymnasium API checker.
"""

import sys
import traceback
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

from environment.edge_environment import EdgeEnvironment
from environment.edge_scheduling_env import (
    EdgeSchedulingEnv,
    EnvConfig,
    rollout_with_scheduler,
)
from environment.state_builder import MaskConfig, ObservationConfig, StateBuilder
from environment.task_generator import Task
from rewards.reward_engine import TERMS, RewardConfig
from scheduling.fifo import FIFOScheduler
from scheduling.greedy_scheduler import GreedyScheduler
from scheduling.random_scheduler import RandomScheduler
from scheduling.round_robin import RoundRobinScheduler

ROOT = Path(__file__).resolve().parent
SERVER_YAML = ROOT / "config" / "server_config.yaml"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

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


def make_env(**environment_overrides):
    """Env from config files; kwargs override keys of the `environment:` section."""
    overrides = {"environment": environment_overrides} if environment_overrides else None
    return EdgeSchedulingEnv.from_config_files(env_overrides=overrides)


def run_masked_random(env, seed, options=None, policy_seed=0):
    """Random policy restricted to the action mask. Returns (last_info, levels, trace)."""
    rng = np.random.default_rng(policy_seed)
    obs, info = env.reset(seed=seed, options=options)
    levels = [info["action_mask_level"]]
    trace = []
    while True:
        valid = np.flatnonzero(env.action_masks())
        assert len(valid) > 0, "action mask must never be empty"
        action = int(rng.choice(valid))
        obs, reward, terminated, truncated, info = env.step(action)
        trace.append((obs, reward, info))
        levels.append(info["action_mask_level"])
        if terminated or truncated:
            return info, levels, trace


# ----------------------------------------------------------------------
# API contract
# ----------------------------------------------------------------------

def test_spaces_and_reset_contract():
    env = make_env()
    obs, info = env.reset(seed=1)
    assert env.action_space.n == 4
    assert obs.shape == env.observation_space.shape == (env.state_builder.obs_dim,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert np.isfinite(obs).all()
    mask = info["action_mask"]
    assert mask.dtype == bool and mask.shape == (4,) and mask.any()
    assert np.array_equal(mask, env.action_masks())
    assert len(env.feature_names) == obs.shape[0]
    assert len(set(env.feature_names)) == len(env.feature_names)  # unique names


def test_reset_is_deterministic_for_a_seed():
    a, b, c = make_env(), make_env(), make_env()
    oa, ia = a.reset(seed=123)
    ob, ib = b.reset(seed=123)
    oc, ic = c.reset(seed=124)
    assert np.array_equal(oa, ob)
    assert (ia["workload_type"], ia["network_scenario"], ia["task_seed"]) == (
        ib["workload_type"], ib["network_scenario"], ib["task_seed"])
    assert ia["task_seed"] != ic["task_seed"]
    # identical action sequences -> identical trajectories, bit for bit
    for action in [0, 1, 2, 3, 3, 2, 1, 0, 1, 1, 2, 3]:
        ra = a.step(action)
        rb = b.step(action)
        assert np.array_equal(ra[0], rb[0]) and ra[1] == rb[1] and ra[2] == rb[2]
        if ra[2] or ra[3]:
            break


def test_reset_options_override_and_validation():
    env = make_env()
    _, info = env.reset(seed=0, options={
        "workload_type": "heavy", "network_scenario": "congested",
        "duration_ms": 3000.0, "task_seed": 5})
    assert (info["workload_type"], info["network_scenario"], info["task_seed"]) == (
        "heavy", "congested", 5)
    assert info["duration_ms"] == 3000.0
    assert all(t.arrival_time_ms < 3000.0 for t in env.sim.tasks)
    with expect_raises(ValueError, "Unknown reset option"):
        env.reset(options={"workload": "heavy"})  # typo


def test_step_before_reset_and_after_end_raises():
    env = make_env()
    with expect_raises(RuntimeError):
        env.step(0)
    run_masked_random(env, seed=0, options={"duration_ms": 1500.0})
    with expect_raises(RuntimeError):
        env.step(0)  # episode already ended
    with expect_raises(ValueError):
        env.reset(seed=0)
        env.step(99)  # out-of-range action


# ----------------------------------------------------------------------
# Correctness against the simulator
# ----------------------------------------------------------------------

def test_baselines_match_run_episode():
    """Driving a baseline through Gym == driving it through run_episode()."""
    env = make_env()
    factories = {
        "FIFO": FIFOScheduler,
        "RoundRobin": RoundRobinScheduler,
        "Random": lambda: RandomScheduler(7),
        "Greedy": GreedyScheduler,
    }
    for name, make in factories.items():
        for workload in ("normal", "heavy", "burst", "variable"):
            for network in ("normal", "congested"):
                opts = {"workload_type": workload, "network_scenario": network, "task_seed": 11}
                ep = rollout_with_scheduler(env, make(), seed=1, options=opts)

                sim = EdgeEnvironment(server_cfgs(), num_users=15, time_step_ms=10.0)
                ref = sim.run_episode(make(), workload, network, 10000.0, seed=11)
                for key, value in ref.items():
                    assert ep[key] == value, (name, workload, network, key, ep[key], value)


def test_task_conservation_and_no_double_counting():
    env = make_env()
    for seed, workload in enumerate(["normal", "heavy", "burst", "variable"]):
        info, _, trace = run_masked_random(
            env, seed=seed, options={"workload_type": workload, "network_scenario": "congested"})
        sim = env.sim
        completed = [c["task_id"] for c in sim.completed_tasks]
        failed = [f["task_id"] for f in sim.failed_tasks]
        all_ids = completed + failed
        # every task ends exactly once, as completed XOR failed
        assert len(all_ids) == len(set(all_ids)) == len(sim.tasks), (workload, len(all_ids), len(sim.tasks))
        assert set(all_ids) == {t.task_id for t in sim.tasks}
        # one decision per task
        assert info["episode"]["num_decisions"] == len(sim.tasks)
        # per-step event counts add up to the simulator's own logs
        assert sum(t[2]["n_completed"] for t in trace) == len(completed)
        assert sum(t[2]["n_rejected"] + t[2]["n_deadline_failures"] for t in trace) == len(failed)


def test_reward_terms_match_simulator_log():
    """Recompute latency / SLA / rejection totals from the simulator's raw logs
    and compare with what the reward accumulated over the episode."""
    env = make_env()
    cfg = env.reward_config
    for seed in range(4):
        info, _, _ = run_masked_random(env, seed=seed, options={"workload_type": "normal"})
        sim = env.sim
        prio = {t.task_id: str(t.priority) for t in sim.tasks}
        pw = cfg.priority_weights
        raw = info["episode"]["reward_terms_raw"]

        exp_latency = sum(min(c["latency_ms"] / cfg.latency_ref_ms, cfg.latency_cap)
                          for c in sim.completed_tasks)
        exp_sla = sum(pw[prio[c["task_id"]]] for c in sim.completed_tasks if not c["met_deadline"])
        exp_sla += sum(pw[prio[f["task_id"]]] for f in sim.failed_tasks
                       if f["reason"] == "deadline_missed_while_running")
        exp_rej = sum(pw[prio[f["task_id"]]] for f in sim.failed_tasks
                      if f["reason"] != "deadline_missed_while_running")
        exp_energy = sum(s.total_energy_joules for s in sim.servers.values()) / cfg.energy_ref_j

        assert np.isclose(raw["latency"], exp_latency)
        assert np.isclose(raw["sla"], exp_sla)
        assert np.isclose(raw["rejection"], exp_rej)
        assert np.isclose(raw["energy"], exp_energy)  # every joule counted once
        # episode return equals the sum of weighted terms
        assert np.isclose(info["episode"]["episode_return"],
                          cfg.reward_scale * sum(info["episode"]["reward_terms_weighted"].values()))


def test_preview_network_delay_matches_simulator_and_restores_state():
    sim = EdgeEnvironment(server_cfgs(), num_users=15, time_step_ms=10.0)
    sim.reset("normal", "congested", 10000.0, seed=3)
    sim.pending_queue = []
    mk = lambda i: Task(task_id=1000 + i, user_id=0, cpu_requirement=0.5, ram_requirement=0.5,
                        data_size_mb=6.0, execution_time_ms=300.0, deadline_ms=9000.0,
                        priority="medium", arrival_time_ms=0.0)
    server = sim.servers[1]
    for i in range(3):  # create some in-flight congestion on server 1
        sim.pending_queue = [mk(i)]
        sim.submit_decision(sim.pending_queue[0], 1)

    probe = mk(99)
    before = dict(sim.network.active_transfers)
    preview = StateBuilder.preview_network_delay_ms(sim, probe, server)
    assert {k: v for k, v in sim.network.active_transfers.items() if v} == \
           {k: v for k, v in before.items() if v}, "preview must not change transfer counts"

    sim.pending_queue = [probe]
    sim.submit_decision(probe, 1)
    actual = sim.pending_transfers[-1]["ready_time_ms"] - sim.current_time_ms
    assert np.isclose(preview, actual), (preview, actual)


# ----------------------------------------------------------------------
# Action mask
# ----------------------------------------------------------------------

def test_masked_policy_is_never_rejected():
    """With mask-respecting actions, no task is ever rejected (server_full /
    queue_full_on_arrival) unless the mask had to fall back (all servers full)."""
    env = make_env()
    rejections = {"server_full_or_incompatible", "queue_full_on_arrival", "invalid_server_id"}
    for seed, workload in enumerate(["normal", "heavy", "burst", "variable", "heavy", "burst"]):
        network = "congested" if seed % 2 else "normal"
        info, levels, trace = run_masked_random(
            env, seed=seed, options={"workload_type": workload, "network_scenario": network},
            policy_seed=seed)
        assert all(t[2]["action_was_valid"] for t in trace)
        if max(levels) < 2:  # mask never had to fall back
            bad = rejections & set(info["episode"]["failure_reasons"])
            assert not bad, (workload, network, info["episode"]["failure_reasons"])
            assert all(t[2]["accepted"] for t in trace)


def test_mask_fallback_keeps_mask_non_empty_when_all_servers_full():
    tiny = [dict(c, max_queue_length=1) for c in server_cfgs()[:2]]
    env = EdgeSchedulingEnv(tiny, EnvConfig(), RewardConfig())
    info, levels, trace = run_masked_random(
        env, seed=0, options={"workload_type": "heavy", "network_scenario": "congested"})
    assert max(levels) >= 2, "expected the fallback path to be exercised"
    assert info["episode"]["num_failed"] > 0  # overload really produced failures
    assert "episode" in info


def test_overload_fraction_prunes_nearly_full_servers():
    cfgs = server_cfgs()
    server3_max_q = next(c["max_queue_length"] for c in cfgs if c["server_id"] == 3)  # 15

    def mask_after(n_inflight, fraction):
        sim = EdgeEnvironment(cfgs, num_users=15, time_step_ms=10.0)
        sim.reset("normal", "normal", 10000.0, seed=0)
        sim.pending_queue = []
        for i in range(n_inflight):
            t = Task(task_id=5000 + i, user_id=0, cpu_requirement=0.5, ram_requirement=0.5,
                     data_size_mb=1.0, execution_time_ms=200.0, deadline_ms=1e9,
                     priority="low", arrival_time_ms=0.0)
            sim.pending_queue = [t]
            sim.submit_decision(t, 3)
        probe = Task(task_id=9999, user_id=0, cpu_requirement=0.5, ram_requirement=0.5,
                     data_size_mb=1.0, execution_time_ms=200.0, deadline_ms=1e9,
                     priority="low", arrival_time_ms=0.0)
        sb = StateBuilder(sim, ObservationConfig(), MaskConfig(overload_fraction=fraction))
        mask, level = sb.action_mask(sim, probe)
        return bool(mask[sb.server_ids.index(3)]), level

    # threshold = 0.5 * 15 = 7.5 -> allowed at 7 in flight, pruned at 8
    assert mask_after(7, 0.5) == (True, 0)
    assert mask_after(8, 0.5) == (False, 0)
    # with fraction 1.0 the same server is still allowed at 8
    assert mask_after(8, 1.0) == (True, 0)
    # completely full server is excluded even at fraction 1.0
    assert mask_after(server3_max_q, 1.0) == (False, 0)


# ----------------------------------------------------------------------
# Observation
# ----------------------------------------------------------------------

def test_observations_stay_in_bounds_and_are_finite_across_workloads():
    env = make_env()
    for seed, workload in enumerate(["normal", "heavy", "burst", "variable"]):
        _, _, trace = run_masked_random(
            env, seed=seed, options={"workload_type": workload, "network_scenario": "congested"})
        for obs, reward, _ in trace:
            assert env.observation_space.contains(obs)
            assert np.isfinite(obs).all() and np.isfinite(reward)


def test_decode_observation_and_static_features():
    env = make_env()
    obs, _ = env.reset(seed=2, options={"workload_type": "normal"})
    d = env.decode_observation(obs)
    assert set(d["servers"]) == set(env.server_ids)
    task = d["task"]
    assert np.isclose(task["task_prio_high"] + task["task_prio_medium"] + task["task_prio_low"], 1.0)
    # the largest fleet capacity normalises to exactly 1
    caps = [d["servers"][sid]["cpu_cores"] for sid in env.server_ids]
    assert max(caps) == 1.0 and min(caps) < 1.0
    # static features do not change between steps
    obs2, *_ = env.step(int(np.flatnonzero(env.action_masks())[0]))
    d2 = env.decode_observation(obs2)
    for sid in env.server_ids:
        for feat in ("cpu_cores", "ram_gb", "bandwidth", "power_max"):
            assert d["servers"][sid][feat] == d2["servers"][sid][feat]
    with expect_raises(ValueError):
        env.decode_observation(np.zeros(3))


def test_include_feasibility_flag_adds_mask_bits_to_observation():
    base = make_env()
    with_feas = EdgeSchedulingEnv.from_config_files(
        env_overrides={"observation": {"include_feasibility": True}})
    assert with_feas.state_builder.obs_dim == base.state_builder.obs_dim + base.action_space.n
    obs, info = with_feas.reset(seed=4, options={"workload_type": "normal"})
    d = with_feas.decode_observation(obs)["servers"]
    bits = [d[sid]["feasible"] for sid in with_feas.server_ids]
    assert info["action_mask_level"] == 0
    assert np.array_equal(np.array(bits, dtype=bool), info["action_mask"])
    assert "feasible" not in base.decode_observation(base.reset(seed=4)[0])["servers"][1]


def test_observation_reflects_load_after_a_decision():
    env = make_env()
    obs, _ = env.reset(seed=8, options={"workload_type": "normal"})
    sid = env.server_ids[0]
    before = env.decode_observation(obs)["servers"][sid]["queue_fill"]
    # put several tasks on server 1 back-to-back and check queue_fill rises
    fills = [before]
    for _ in range(3):
        if not env.sim.has_pending_decision():
            break
        obs, *_ = env.step(0)
        fills.append(env.decode_observation(obs)["servers"][sid]["queue_fill"])
    assert max(fills) > before


# ----------------------------------------------------------------------
# Safety guards, ablations, scaling
# ----------------------------------------------------------------------

def test_unseen_conditions_are_blocked_unless_explicitly_allowed():
    with expect_raises(ValueError, "unseen"):
        make_env(workloads=["normal", "unseen"])
    with expect_raises(ValueError, "unseen"):
        make_env(network_scenarios=["unseen"])
    env = make_env()
    with expect_raises(ValueError, "unseen"):
        env.reset(options={"workload_type": "unseen"})
    with expect_raises(ValueError, "unseen"):
        env.reset(options={"network_scenario": "unseen"})

    ev = make_env(allow_unseen=True)  # evaluation env: allowed, and obs stay in bounds
    ev.reset(seed=0, options={"workload_type": "unseen", "network_scenario": "unseen"})
    while True:
        valid = np.flatnonzero(ev.action_masks())
        obs, r, term, trunc, _ = ev.step(int(valid[0]))
        assert ev.observation_space.contains(obs) and np.isfinite(r)
        if term or trunc:
            break


def test_unknown_config_keys_are_rejected():
    with expect_raises(ValueError, "Unknown environment config key"):
        make_env(worklodas=["normal"])
    with expect_raises(ValueError, "Unknown observation config key"):
        EdgeSchedulingEnv.from_config_files(env_overrides={"observation": {"bogus": 1}})
    with expect_raises(ValueError, "Unknown env config section"):
        EnvConfig.from_dict({"enviroment": {}})


def test_reward_ablation_via_config_overrides():
    opts = {"workload_type": "normal", "task_seed": 3}
    full = EdgeSchedulingEnv.from_config_files()
    abl = EdgeSchedulingEnv.from_config_files(reward_overrides={"weights": {"energy": 0.0}})
    ep_full = rollout_with_scheduler(full, GreedyScheduler(), seed=0, options=opts)
    ep_abl = rollout_with_scheduler(abl, GreedyScheduler(), seed=0, options=opts)
    assert ep_abl["reward_terms_weighted"]["energy"] == 0.0
    assert ep_abl["reward_terms_raw"]["energy"] == ep_full["reward_terms_raw"]["energy"] > 0.0
    assert ep_abl["episode_return"] > ep_full["episode_return"]
    # same decisions -> same simulator outcome regardless of reward weights
    assert ep_abl["num_completed"] == ep_full["num_completed"]


def test_max_ticks_truncates_instead_of_hanging():
    def run_until_end(env):
        for _ in range(100_000):
            _, _, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                return terminated, truncated, info
        raise AssertionError("episode never ended")

    # limit hit while still fast-forwarding to the first task (inside reset)
    env = make_env(max_ticks=1)
    env.reset(seed=0, options={"workload_type": "normal"})
    terminated, truncated, info = run_until_end(env)
    assert truncated and not terminated and info["episode"]["truncated"] is True

    # limit hit in the middle of an episode
    env = make_env(max_ticks=300)
    env.reset(seed=0, options={"workload_type": "normal"})
    terminated, truncated, info = run_until_end(env)
    assert truncated and not terminated and info["episode"]["truncated"] is True
    assert 0 < info["episode"]["num_decisions"] < len(env.sim.tasks)

    # a normal episode is NOT truncated
    env = make_env()
    env.reset(seed=0, options={"workload_type": "normal"})
    terminated, truncated, info = run_until_end(env)
    assert terminated and not truncated and info["episode"]["truncated"] is False


def test_scales_to_more_servers():
    base = server_cfgs()
    for n in (2, 8, 16):
        cfgs = []
        for i in range(n):
            c = dict(base[i % len(base)])
            c["server_id"] = i + 1
            cfgs.append(c)
        env = EdgeSchedulingEnv(cfgs, EnvConfig(), RewardConfig())
        assert env.action_space.n == n
        assert env.observation_space.shape == (env.state_builder.obs_dim,)
        info, _, trace = run_masked_random(env, seed=1, options={"workload_type": "normal"})
        assert "episode" in info and len(trace) > 0


def test_env_never_mutates_baselines_or_config_files():
    """Building/using the env must leave module-level presets untouched."""
    from environment.task_generator import WORKLOAD_PRESETS
    snapshot = repr(sorted(WORKLOAD_PRESETS.items()))
    env = make_env()
    run_masked_random(env, seed=0, options={"workload_type": "burst"})
    assert repr(sorted(WORKLOAD_PRESETS.items())) == snapshot


def test_gymnasium_check_env():
    """The official Gymnasium API conformance checker."""
    try:
        from gymnasium.utils.env_checker import check_env
    except ImportError as e:
        raise unittest.SkipTest(f"gymnasium env_checker not available here: {e}")
    check_env(make_env(), skip_render_check=True)


# ----------------------------------------------------------------------
# tiny runner so this file also works as `python test_drl_env.py`
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
    print(f"\n{len(tests) - failures}/{len(tests)} passed/skipped" + ("" if not failures else f", {failures} FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())