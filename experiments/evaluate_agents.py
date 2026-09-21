"""
experiments/evaluate_agents.py

Phase 7: head-to-head evaluation of the trained masked-PPO scheduler against
every non-learning baseline (FIFO, RoundRobin, Random, Greedy).

Usage:
    python -m experiments.evaluate_agents
    python -m experiments.evaluate_agents --include-unseen
    python -m experiments.evaluate_agents --model models/ppo_checkpoints/final_model
    python -m experiments.evaluate_agents --episodes-per-condition 20
    python -m experiments.evaluate_agents --baselines-only      (no model needed)

Design (so the numbers can be trusted)
--------------------------------------
* BALANCED GRID: every (workload x network) pair gets the same number of
  episodes, with the condition pinned explicitly - never left to a random draw.
* PAIRED: episode i of a condition uses the same task_seed for every agent, so
  all agents face the identical task stream. Differences between agents are
  therefore differences in decisions, not in luck.
* HELD-OUT SEEDS: seeds start at 50_000, disjoint from training and from the
  seeds used by the mid-training eval callback (10_000+).
* DETERMINISTIC POLICY + ACTION MASKS at inference, exactly as in training-time eval.
* --include-unseen also runs the `unseen` workload / network conditions
  (generalisation test). These are reported separately from the seen conditions.
* Uncertainty: the headline PPO-vs-Greedy comparison reports the mean paired
  return difference with a bootstrap 95% CI and a win rate.

Outputs (results/evaluation/):
    episodes.csv               one row per (agent, condition, episode)
    summary_by_condition.csv   mean/std per (agent, condition)
    summary_overall.csv        macro-average per (agent, seen/unseen)
    paired_vs_baselines.csv    PPO minus each baseline: mean diff, CI, win rate
"""

from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from environment.edge_scheduling_env import EdgeSchedulingEnv, rollout_with_scheduler
from scheduling.ect_scheduler import EarliestCompletionScheduler
from scheduling.fifo import FIFOScheduler
from scheduling.greedy_scheduler import GreedyScheduler
from scheduling.random_scheduler import RandomScheduler
from scheduling.round_robin import RoundRobinScheduler
from training.train_ppo import training_conditions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "ppo_checkpoints" / "best_model"
DEFAULT_OUT = PROJECT_ROOT / "results" / "evaluation"
EVAL_SEED_BASE = 50_000

BASELINE_FACTORIES = {
    "FIFO": lambda seed: FIFOScheduler(),
    "RoundRobin": lambda seed: RoundRobinScheduler(),
    "Random": lambda seed: RandomScheduler(seed),
    "Greedy": lambda seed: GreedyScheduler(),
    "ECT": lambda seed: EarliestCompletionScheduler(),
}
PPO = "PPO"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def flatten_summary(summary: dict) -> dict:
    """Keep scalar fields; expand reward-term and failure-reason dicts."""
    row: dict = {}
    for key, value in summary.items():
        if isinstance(value, (bool, np.bool_)):
            row[key] = float(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            row[key] = float(value)
        elif key in ("reward_terms_raw", "reward_terms_weighted") and isinstance(value, dict):
            prefix = "raw" if key.endswith("raw") else "w"
            for term, x in value.items():
                row[f"{prefix}_{term}"] = float(x)
        elif key == "failure_reasons" and isinstance(value, dict):
            for reason, n in value.items():
                row[f"fail_{reason}"] = float(n)
    return row


def run_ppo_episode(env, model, seed: int, options: dict, deterministic: bool = True) -> dict:
    obs, _ = env.reset(seed=seed, options=options)
    done = False
    info: dict = {}
    while not done:
        action, _ = model.predict(
            obs, action_masks=env.action_masks(), deterministic=deterministic
        )
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    return info["episode_summary"]


def build_conditions(env, include_unseen: bool) -> list[tuple[str, str, bool]]:
    """(workload, network, is_seen) for every pair in the grid."""
    workloads, networks = training_conditions(env)
    seen = set(product(workloads, networks))
    workloads, networks = list(workloads), list(networks)
    if include_unseen:
        workloads.append("unseen")
        networks.append("unseen")
    return [(w, n, (w, n) in seen) for w, n in product(workloads, networks)]


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate PPO vs baselines on a balanced, paired grid.")
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                    help="Path to a MaskablePPO .zip (with or without the .zip suffix).")
    ap.add_argument("--episodes-per-condition", type=int, default=10)
    ap.add_argument("--include-unseen", action="store_true",
                    help="Also evaluate the held-out 'unseen' workload/network conditions.")
    ap.add_argument("--stochastic", action="store_true",
                    help="Sample PPO actions instead of taking the argmax.")
    ap.add_argument("--baselines-only", action="store_true", help="Skip PPO.")
    ap.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_overrides = {"environment": {"allow_unseen": True}} if args.include_unseen else None
    env = EdgeSchedulingEnv.from_config_files(env_overrides=env_overrides)
    conditions = build_conditions(env, args.include_unseen)

    # Fail fast: make sure every condition can actually be reset BEFORE spending
    # minutes on rollouts.
    for w, n, _ in conditions:
        env.reset(seed=0, options={"workload_type": w, "network_scenario": n})

    model = None
    agents = list(BASELINE_FACTORIES)
    if not args.baselines_only:
        zip_path = Path(str(args.model) if str(args.model).endswith(".zip") else f"{args.model}.zip")
        if not zip_path.exists():
            raise FileNotFoundError(
                f"{zip_path} not found. Train first (python -m training.train_ppo) "
                "or pass --baselines-only."
            )
        model = MaskablePPO.load(str(zip_path), device="cpu")
        agents.append(PPO)
        print(f"Loaded PPO model: {zip_path}")

    total = len(conditions) * args.episodes_per_condition * len(agents)
    print(f"{len(conditions)} conditions x {args.episodes_per_condition} episodes x "
          f"{len(agents)} agents = {total} rollouts")

    # ---- run everything ----
    rows = []
    t0 = time.time()
    for w, n, seen in conditions:
        for i in range(args.episodes_per_condition):
            seed = args.seed_base + i
            opts = {"workload_type": w, "network_scenario": n, "task_seed": seed}
            for agent in agents:
                if agent == PPO:
                    summary = run_ppo_episode(env, model, seed, opts, not args.stochastic)
                else:
                    summary = rollout_with_scheduler(
                        env, BASELINE_FACTORIES[agent](seed), seed=seed, options=opts
                    )
                rows.append({
                    "agent": agent, "workload": w, "network": n, "seen": seen,
                    "condition": f"{w}_{n}", "episode": i, "seed": seed,
                    **flatten_summary(summary),
                })
        print(f"  done {w}/{n:<9} ({'seen' if seen else 'UNSEEN'})  "
              f"[{time.time() - t0:.0f}s elapsed]")

    df = pd.DataFrame(rows)
    fail_cols = [c for c in df.columns if c.startswith("fail_")]
    df[fail_cols] = df[fail_cols].fillna(0.0)
    df.to_csv(out_dir / "episodes.csv", index=False)

    # ---- summaries ----
    metrics = [c for c in ("episode_return", "completion_rate", "num_completed", "num_failed")
               if c in df.columns] + [c for c in df.columns if c.startswith("w_")]

    by_cond = (df.groupby(["agent", "seen", "condition"])[metrics]
                 .agg(["mean", "std"]))
    by_cond.columns = [f"{m}_{s}" for m, s in by_cond.columns]
    by_cond.to_csv(out_dir / "summary_by_condition.csv")

    # macro-average: mean over conditions of the per-condition mean (equal weight)
    cond_means = df.groupby(["agent", "seen", "condition"])[metrics].mean().reset_index()
    overall = cond_means.groupby(["agent", "seen"])[metrics].mean()
    overall.to_csv(out_dir / "summary_overall.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

    print("\n=== Mean episode_return by condition (higher is better) ===")
    print(df.pivot_table(index="condition", columns="agent", values="episode_return",
                         aggfunc="mean")[agents])
    print("\n=== Mean completion_rate by condition ===")
    print(df.pivot_table(index="condition", columns="agent", values="completion_rate",
                         aggfunc="mean")[agents])
    print("\n=== Overall (macro-average across conditions) ===")
    print(overall[[m for m in ("episode_return", "completion_rate", "num_failed") if m in overall]])

    # ---- paired comparison: PPO vs each baseline ----
    if model is not None:
        wide = df.pivot_table(index=["seen", "condition", "episode"], columns="agent",
                              values="episode_return")
        paired_rows = []
        for baseline in BASELINE_FACTORIES:
            for label, mask in (("seen", wide.index.get_level_values("seen") == True),   # noqa: E712
                                ("unseen", wide.index.get_level_values("seen") == False)):  # noqa: E712
                if not mask.any():
                    continue
                diff = (wide[PPO] - wide[baseline])[mask].to_numpy()
                lo, hi = bootstrap_ci(diff)
                paired_rows.append({
                    "baseline": baseline, "conditions": label, "n_episodes": len(diff),
                    "mean_diff": diff.mean(), "ci95_low": lo, "ci95_high": hi,
                    "win_rate": float((diff > 0).mean()),
                    "significant": bool(lo > 0 or hi < 0),
                })
        paired = pd.DataFrame(paired_rows)
        paired.to_csv(out_dir / "paired_vs_baselines.csv", index=False)
        print("\n=== PPO minus baseline: paired episode_return (positive = PPO better) ===")
        print(paired.to_string(index=False))

        g = paired[paired["baseline"] == "Greedy"]
        print("\nVerdict vs Greedy:")
        for _, r in g.iterrows():
            if r["ci95_low"] > 0:
                word = "BEATS"
            elif r["ci95_high"] < 0:
                word = "LOSES TO"
            else:
                word = "is statistically tied with"
            print(f"  {r['conditions']:>6}: PPO {word} Greedy "
                  f"(mean diff {r['mean_diff']:+.2f}, 95% CI [{r['ci95_low']:+.2f}, "
                  f"{r['ci95_high']:+.2f}], wins {r['win_rate'] * 100:.0f}%)")

    print(f"\nWrote CSVs to {out_dir}")


if __name__ == "__main__":
    main()