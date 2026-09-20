"""
experiments/failure_breakdown.py

Shows WHY tasks fail for each agent, per condition, from the CSV written by
experiments/evaluate_agents.py.

    python -m experiments.failure_breakdown
    python -m experiments.failure_breakdown --agents Greedy PPO RoundRobin
"""
import argparse
from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parents[1] / "results" / "evaluation" / "episodes.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--agents", nargs="+", default=["Greedy", "PPO"])
args = ap.parse_args()

df = pd.read_csv(CSV)
fail_cols = [c for c in df.columns if c.startswith("fail_")]
if not fail_cols:
    raise SystemExit("No fail_* columns in episodes.csv (no failures recorded?).")
df[fail_cols] = df[fail_cols].fillna(0.0)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)

mean_fail = df.groupby(["condition", "agent"])[fail_cols].mean()
mean_fail.columns = [c.removeprefix("fail_") for c in mean_fail.columns]

# Also show completed / total decisions so counts have context.
extra = df.groupby(["condition", "agent"])[
    [c for c in ("num_completed", "num_failed", "num_decisions") if c in df.columns]
].mean()

table = extra.join(mean_fail).round(1)
table = table[table.index.get_level_values("agent").isin(args.agents)]

print("Mean tasks per episode, by failure reason:\n")
print(table.to_string())

# Share of failures by reason, pooled over all conditions.
share = (df[df["agent"].isin(args.agents)].groupby("agent")[fail_cols].sum())
share = share.div(share.sum(axis=1), axis=0) * 100
share.columns = [c.removeprefix("fail_") for c in share.columns]
print("\nShare of all failures by reason (%), pooled over conditions:\n")
print(share.round(1).to_string())