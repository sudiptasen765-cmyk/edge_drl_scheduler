"""
compare_heuristics.py

Compare the rule-based schedulers (FIFO, RoundRobin, Random, Greedy and the
new Earliest-Completion-Time heuristic) on identical, seeded episodes, with
paired bootstrap confidence intervals.

Purpose: find out how much better than Greedy a non-learning scheduler can do.
That is the "achievable headroom" a DRL agent should at least reach.

    python -m experiments.compare_heuristics
    python -m experiments.compare_heuristics --seeds 30 --include-unseen
    python -m experiments.compare_heuristics --a ECT --b Greedy

Everything is driven through the Gymnasium environment with the project's
reward, so the numbers are directly comparable to the DRL results. The script
uses only the environment's public API (reset / step / sim / server_ids) and
adds up the rewards itself.

Outputs (in --out-dir, default results/heuristics):
    episodes.csv   one row per (scheduler, condition, seed)
    summary.csv    paired A-minus-B comparison per condition and overall
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.edge_scheduling_env import EdgeSchedulingEnv  # noqa: E402
from scheduling.ect_scheduler import EarliestCompletionScheduler  # noqa: E402
from scheduling.fifo import FIFOScheduler  # noqa: E402
from scheduling.greedy_scheduler import GreedyScheduler  # noqa: E402
from scheduling.random_scheduler import RandomScheduler  # noqa: E402
from scheduling.round_robin import RoundRobinScheduler  # noqa: E402

SCHEDULER_NAMES = ["FIFO", "RoundRobin", "Random", "Greedy", "ECT"]


def make_scheduler(name: str, seed: int):
    """Fresh instance per episode so no state leaks between episodes."""
    return {
        "FIFO": FIFOScheduler,
        "RoundRobin": RoundRobinScheduler,
        "Random": lambda: RandomScheduler(seed),
        "Greedy": GreedyScheduler,
        "ECT": EarliestCompletionScheduler,
    }[name]()


def run_episode(env: EdgeSchedulingEnv, scheduler, cond: dict) -> dict:
    """One full episode; returns the episode's metrics."""
    env.reset(seed=cond["task_seed"], options=cond)
    total_reward = 0.0
    while True:
        task = env.sim.next_pending_task()
        action = 0 if task is None else env.server_ids.index(scheduler(task, env.sim))
        _obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    sim = env.sim
    n = max(len(sim.tasks), 1)
    completed = sim.completed_tasks
    return {
        "num_tasks": len(sim.tasks),
        "episode_return": total_reward,
        "return_per_task": total_reward / n,
        "on_time_rate": sum(1 for c in completed if c["met_deadline"]) / n,
        "completion_rate": len(completed) / n,
        "avg_latency_ms": float(np.mean([c["latency_ms"] for c in completed])) if completed else 0.0,
        "energy_per_task_j": sum(s.total_energy_joules for s in sim.servers.values()) / n,
    }


def paired_bootstrap(diffs: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Mean of paired differences with a 95% bootstrap interval."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def verdict(lo: float, hi: float) -> str:
    return "A BETTER" if lo > 0 else "A WORSE" if hi < 0 else "tie"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare rule-based schedulers on identical episodes.")
    ap.add_argument("--workloads", nargs="+", default=["normal", "variable", "heavy", "burst"])
    ap.add_argument("--networks", nargs="+", default=["normal", "congested"])
    ap.add_argument("--seeds", type=int, default=20, help="episodes per (workload, network)")
    ap.add_argument("--seed-base", type=int, default=700_000_000)
    ap.add_argument("--include-unseen", action="store_true",
                    help="also evaluate the 'unseen' workload and network (evaluation only)")
    ap.add_argument("--a", default="ECT", choices=SCHEDULER_NAMES, help="scheduler A of the paired comparison")
    ap.add_argument("--b", default="Greedy", choices=SCHEDULER_NAMES, help="scheduler B of the paired comparison")
    ap.add_argument("--out-dir", default="results/heuristics")
    args = ap.parse_args(argv)

    workloads, networks = list(args.workloads), list(args.networks)
    if args.include_unseen:
        workloads = workloads + ["unseen"]
        networks = networks + ["unseen"]
    overrides = {"environment": {"allow_unseen": True}} if args.include_unseen else None
    env = EdgeSchedulingEnv.from_config_files(env_overrides=overrides)

    rows = []
    seed = args.seed_base
    total = len(workloads) * len(networks) * args.seeds
    done = 0
    for workload in workloads:
        for network in networks:
            for _ in range(args.seeds):
                cond = {"workload_type": workload, "network_scenario": network, "task_seed": seed}
                for name in SCHEDULER_NAMES:
                    m = run_episode(env, make_scheduler(name, seed), cond)
                    rows.append({"scheduler": name, "workload": workload, "network": network,
                                 "task_seed": seed, **m})
                seed += 1
                done += 1
            print(f"  finished {workload}/{network}  ({done}/{total} conditions x seeds)", flush=True)

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "episodes.csv", index=False)

    # ---- table 1: mean return per task by scheduler and condition ----
    df["condition"] = df["workload"] + " | " + df["network"]
    table1 = (df.pivot_table(index="condition", columns="scheduler", values="return_per_task", aggfunc="mean")
                [SCHEDULER_NAMES].round(3))
    print("\nMean return per task (higher is better)\n")
    print(table1.to_string())

    # ---- table 2: paired A - B ----
    a, b = args.a, args.b
    pa = df[df.scheduler == a].set_index(["condition", "task_seed"])
    pb = df[df.scheduler == b].set_index(["condition", "task_seed"])
    summary = []
    for cond_name, idx in list(pa.groupby(level=0).groups.items()) + [("ALL CONDITIONS", None)]:
        sel_a = pa if idx is None else pa.loc[idx]
        sel_b = pb.loc[sel_a.index]
        rec = {"condition": cond_name, "n": len(sel_a)}
        for metric, label in (("episode_return", "return"), ("on_time_rate", "on_time")):
            d = (sel_a[metric] - sel_b[metric]).to_numpy()
            mean, lo, hi = paired_bootstrap(d)
            rec[f"{label}_diff"], rec[f"{label}_lo"], rec[f"{label}_hi"] = mean, lo, hi
            if metric == "episode_return":
                rec["win_rate"] = float((d > 0).mean())
                rec["verdict"] = verdict(lo, hi)
        summary.append(rec)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(out_dir / "summary.csv", index=False)

    print(f"\n{a} minus {b}: paired episode_return, 95% bootstrap CI (positive = {a} better)\n")
    show = pd.DataFrame({
        "condition": sdf["condition"],
        "n": sdf["n"],
        "return diff [95% CI]": [f"{r.return_diff:+.1f} [{r.return_lo:+.1f}, {r.return_hi:+.1f}]" for r in sdf.itertuples()],
        "on-time diff": [f"{r.on_time_diff:+.3f}" for r in sdf.itertuples()],
        "win rate": sdf["win_rate"].round(2),
        "verdict": sdf["verdict"].str.replace("A", a, regex=False),
    })
    print(show.to_string(index=False))
    print(f"\nSaved {out_dir / 'episodes.csv'} and {out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())