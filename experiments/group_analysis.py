"""
experiments/group_analysis.py

Regroups evaluation results by what the models were ACTUALLY trained on, since
the 'seen' label written by evaluate_agents.py only means "in the default env
config".  Reads each folder's episodes.csv (no re-evaluation needed).

Groups (defaults match: --train-workloads normal variable, both networks):
  A  trained conditions        workload in trained set AND network in trained networks
  B  held-out default-config   in the default env config but NOT trained on
  C  unseen network            network == unseen (workload != unseen)
  D  unseen workload           workload == unseen

    python -m experiments.group_analysis "results/evaluation_final_s*"
    python -m experiments.group_analysis "results/evaluation_allload_s*" \
        --trained-workloads normal variable heavy burst --trained-networks normal
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--trained-workloads", nargs="+", default=["normal", "variable"])
ap.add_argument("--trained-networks", nargs="+", default=["normal", "congested"])
ap.add_argument("--default-workloads", nargs="+", default=["normal", "heavy", "burst", "variable"])
ap.add_argument("--default-networks", nargs="+", default=["normal", "congested"])
ap.add_argument("--baseline", default="Greedy")
ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "figures" / "group_analysis.csv"))
args = ap.parse_args()

TRAINED = set(args.trained_workloads)
TRAINED_N = set(args.trained_networks)
DEFAULT_W = set(args.default_workloads)
DEFAULT_N = set(args.default_networks)
B = args.baseline


def group_of(w: str, n: str) -> str:
    if w == "unseen":
        return "D unseen workload"
    if n == "unseen":
        return "C unseen network"
    if w in TRAINED and n in TRAINED_N:
        return "A trained conditions"
    if w in DEFAULT_W and n in DEFAULT_N:
        return "B held-out (default cfg)"
    return "D unseen workload"


def boot_ci(x, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


dirs = []
for pat in args.dirs:
    dirs += sorted(glob.glob(pat)) or [pat]
dirs = [d for d in dirs if (Path(d) / "episodes.csv").exists()]
if not dirs:
    raise SystemExit("No folders with episodes.csv found.")

rows = []
for d in dirs:
    run = Path(d).name.replace("evaluation_", "")
    e = pd.read_csv(Path(d) / "episodes.csv")
    wide = e.pivot_table(index=["workload", "network", "episode"], columns="agent", values="episode_return")
    diff = (wide["PPO"] - wide[B]).reset_index(name="diff")
    diff["group"] = [group_of(w, n) for w, n in zip(diff["workload"], diff["network"])]
    for g, x in diff.groupby("group"):
        v = x["diff"].to_numpy()
        lo, hi = boot_ci(v)
        rows.append({"run": run, "group": g, "n_conditions": x[["workload", "network"]].drop_duplicates().shape[0],
                     "n_episodes": len(v), "mean_diff": v.mean(), "ci_low": lo, "ci_high": hi,
                     "win_rate": (v > 0).mean()})

t = pd.DataFrame(rows)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
t.to_csv(args.out, index=False)

t["cell"] = t.apply(lambda r: f"{r['mean_diff']:+6.1f} [{r['ci_low']:+.1f},{r['ci_high']:+.1f}]", axis=1)
pivot = t.pivot(index="group", columns="run", values="cell")
pivot["mean over runs"] = t.groupby("group")["mean_diff"].mean().map(lambda v: f"{v:+.1f}")
pivot.insert(0, "conds", t.groupby("group")["n_conditions"].first())

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print(f"PPO minus {B}: mean paired episode_return, 95% bootstrap CI  (positive = PPO better)\n")
print(pivot.to_string())
print(f"\nSaved {args.out}")