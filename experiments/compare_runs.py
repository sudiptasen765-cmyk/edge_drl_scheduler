"""
experiments/compare_runs.py

Summarises PPO-vs-Greedy across several evaluation output folders (e.g. one per
training seed), from each folder's paired_vs_baselines.csv / summary_overall.csv.

    python -m experiments.compare_runs results/evaluation_focused_s*
    python -m experiments.compare_runs results/evaluation results/evaluation_focused
"""
import argparse
import glob
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+", help="Evaluation folders or glob patterns.")
ap.add_argument("--baseline", default="Greedy")
args = ap.parse_args()

dirs = []
for pattern in args.dirs:               # PowerShell does not expand globs itself
    dirs += sorted(glob.glob(pattern)) or [pattern]

rows = []
for d in dirs:
    f = Path(d) / "paired_vs_baselines.csv"
    if not f.exists():
        print(f"skip {d}: no paired_vs_baselines.csv")
        continue
    p = pd.read_csv(f)
    p = p[p["baseline"] == args.baseline]
    for _, r in p.iterrows():
        verdict = "BEATS" if r["ci95_low"] > 0 else "LOSES" if r["ci95_high"] < 0 else "tied"
        rows.append({"run": Path(d).name, "conditions": r["conditions"],
                     "mean_diff": r["mean_diff"], "ci95_low": r["ci95_low"],
                     "ci95_high": r["ci95_high"], "win_rate": r["win_rate"], "verdict": verdict})

if not rows:
    raise SystemExit("Nothing to compare.")
t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
print(f"PPO minus {args.baseline}, paired episode_return (positive = PPO better)\n")
print(t.to_string(index=False))
print("\nAcross runs:")
print(t.groupby("conditions").agg(runs=("run", "count"), mean_of_mean_diff=("mean_diff", "mean"),
                                   worst=("mean_diff", "min"), best=("mean_diff", "max"),
                                   runs_beating=("verdict", lambda v: (v == "BEATS").sum())).to_string())