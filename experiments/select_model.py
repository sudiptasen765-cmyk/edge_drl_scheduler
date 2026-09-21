"""
experiments/select_model.py

Pick ONE model to ship from several seeds, using a SELECTION evaluation that is
disjoint from the REPORTING evaluation.

  selection : results/evaluation_select_s<seed>/episodes.csv  (task seeds 50000+)
  reporting : results/evaluation_final_s<seed>/episodes.csv   (task seeds 90000+)

Score = mean over ALL conditions (equal weight) of (PPO - Greedy) episode_return.
Because the reporting seeds were never used to choose, the chosen model's
numbers there are not inflated by the choice.

    python -m experiments.select_model
    python -m experiments.select_model --seeds 1 2 3 4 5 6 7 8
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 9)))
ap.add_argument("--select-prefix", default="evaluation_select")
ap.add_argument("--report-prefix", default="evaluation_final")
ap.add_argument("--results-dir", default=str(ROOT / "results"))
args = ap.parse_args()


def gap(csv):
    d = pd.read_csv(csv)
    m = d.groupby(["condition", "agent"])["episode_return"].mean().unstack("agent")
    return float((m["PPO"] - m["Greedy"]).mean())


rows = []
for s in args.seeds:
    sel = Path(args.results_dir) / f"{args.select_prefix}_s{s}" / "episodes.csv"
    rep = Path(args.results_dir) / f"{args.report_prefix}_s{s}" / "episodes.csv"
    if not sel.exists():
        print(f"(missing {sel})")
        continue
    rows.append({"seed": s, "select_gap": gap(sel),
                 "report_gap": gap(rep) if rep.exists() else float("nan")})
if not rows:
    raise SystemExit("No selection results found.")
r = pd.DataFrame(rows).sort_values("select_gap", ascending=False)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
print("PPO minus Greedy, all-condition macro average (higher is better)\n")
print(r.to_string(index=False))
best = int(r.iloc[0]["seed"])
print(f"\nSelected: seed {best}  -> models/ppo_checkpoints/focused_s{best}/best_model.zip")
print("Selection used task seeds 50000+; its reporting number above is from held-out seeds 90000+.")
print("Report the distribution over ALL seeds as the headline; the selected model is the one to deploy.")