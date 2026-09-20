"""
experiments/reward_breakdown.py

Decomposes the return of each agent into its weighted reward terms
(latency / sla / rejection / energy / ...) using results/evaluation/episodes.csv,
and shows PPO - Greedy per term. Conditions are grouped automatically into
"saturated" (Greedy completes < --sat-threshold of tasks: almost nothing can be
won) and "headroom" (where scheduling quality matters).

    python -m experiments.reward_breakdown
    python -m experiments.reward_breakdown --csv results/evaluation_focused/episodes.csv
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=str(ROOT / "results" / "evaluation" / "episodes.csv"))
ap.add_argument("--sat-threshold", type=float, default=0.15)
ap.add_argument("--a", default="PPO")
ap.add_argument("--b", default="Greedy")
args = ap.parse_args()

df = pd.read_csv(args.csv)
terms = [c for c in df.columns if c.startswith("w_")]
if not terms:
    raise SystemExit("No w_* (weighted reward term) columns found in the CSV.")
cols = ["episode_return", "num_completed"] + terms

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.1f}")

ref = df[df["agent"] == args.b].groupby("condition")["completion_rate"].mean()
saturated = set(ref[ref < args.sat_threshold].index)
df["regime"] = df["condition"].map(lambda c: "saturated" if c in saturated else "headroom")

m = df.groupby(["condition", "agent"])[cols].mean()
a, b = m.xs(args.a, level="agent"), m.xs(args.b, level="agent")
delta = (a - b)
delta["regime"] = delta.index.map(lambda c: "saturated" if c in saturated else "headroom")

print(f"{args.a} minus {args.b}, per condition (negative return diff = {args.a} worse).")
print("Term columns are weighted reward contributions (more negative = costlier).\n")
print(delta.sort_values(["regime", "episode_return"]).to_string())

print(f"\nPooled by regime ({args.a} - {args.b}):\n")
print(delta.groupby("regime")[cols].mean().to_string())

print("\nAbsolute mean contribution per agent by regime:\n")
print(df.groupby(["regime", "agent"])[cols].mean().to_string())