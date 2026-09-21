"""
experiments/seed_summary.py

One-screen view of PPO vs a reference agent (default Greedy) across setups and
seeds, split into the same regimes as reward_breakdown.py ("saturated" = the
reference completes < --sat-threshold of tasks in that condition).

    python -m experiments.seed_summary
    python -m experiments.seed_summary --setups allload focused --seeds 1 2 3
    python -m experiments.seed_summary --b RoundRobin

Reads results/evaluation_<setup>_s<seed>/episodes.csv. Every number is
"PPO minus reference", averaged over conditions with equal weight (as
reward_breakdown.py does). drain_ratio = PPO final_time_ms / reference
final_time_ms (>1 = PPO takes longer to empty the fleet after arrivals stop).
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--setups", nargs="+", default=["allload", "focused"])
ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
ap.add_argument("--a", default="PPO")
ap.add_argument("--b", default="Greedy")
ap.add_argument("--sat-threshold", type=float, default=0.15)
ap.add_argument("--results-dir", default=str(ROOT / "results"))
args = ap.parse_args()

need = ["agent", "condition", "task_seed", "episode_return", "completion_rate",
        "final_time_ms", "w_sla", "w_balance", "w_energy"]
rows = []
for setup in args.setups:
    for seed in args.seeds:
        path = Path(args.results_dir) / f"evaluation_{setup}_s{seed}" / "episodes.csv"
        if not path.exists():
            print(f"(missing {path})")
            continue
        df = pd.read_csv(path)
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise SystemExit(f"{path} lacks columns {miss}")
        ref_c = df[df.agent == args.b].groupby("condition")["completion_rate"].mean()
        sat = set(ref_c[ref_c < args.sat_threshold].index)
        df["regime"] = df.condition.map(lambda c: "saturated" if c in sat else "headroom")
        for regime, g in df.groupby("regime"):
            m = g.groupby(["condition", "agent"])[need[3:]].mean()
            a, b = m.xs(args.a, level="agent"), m.xs(args.b, level="agent")
            d = (a - b).mean()
            pa = g[g.agent == args.a].set_index(["condition", "task_seed"])["episode_return"]
            pb = g[g.agent == args.b].set_index(["condition", "task_seed"])["episode_return"]
            pa, pb = pa.align(pb, join="inner")
            rows.append({
                "setup": setup, "seed": seed, "regime": regime,
                "d_return": d["episode_return"], "d_sla": d["w_sla"],
                "d_balance": d["w_balance"], "d_energy": d["w_energy"],
                "drain_ratio": a["final_time_ms"].mean() / b["final_time_ms"].mean(),
                "win_rate": float((pa > pb).mean()),
                "n_ep": len(pa),
            })

if not rows:
    raise SystemExit("No result folders found.")
r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
print(f"{args.a} minus {args.b}  (win_rate = fraction of paired episodes where {args.a} scores higher)\n")
print(r.sort_values(["regime", "setup", "seed"]).to_string(index=False))
print("\nMean and std across seeds:\n")
agg = r.groupby(["regime", "setup"])[["d_return", "d_sla", "d_balance", "d_energy", "drain_ratio", "win_rate"]].agg(["mean", "std"])
print(agg.to_string())