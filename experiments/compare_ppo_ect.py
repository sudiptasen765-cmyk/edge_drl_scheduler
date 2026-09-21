"""
experiments/compare_ppo_ect.py - PPO vs ECT (and Greedy) on IDENTICAL episodes.

The baselines do not depend on the model, so ONE baselines-only run provides
ECT/Greedy rows for the same task seeds that every PPO evaluation used:

  1. python apply_ect_patch.py   (from the project root; adds ECT to evaluate_agents.py)
  2. python -m experiments.evaluate_agents --baselines-only --include-unseen ^
         --seed-base 90000 --episodes-per-condition 20 --out results/evaluation_baselines_90000
  3. python -m experiments.compare_ppo_ect

For each PPO seed s it pairs results/evaluation_<setup>_s<s>/episodes.csv (PPO rows)
with the baseline file on (condition, seed). Before comparing it CHECKS that the
Greedy rows in both files are identical - if they are not, the two runs did not
face the same episodes and the comparison is refused.

Reported gaps are PPO minus X, macro-averaged over conditions with equal weight
(higher = PPO better), then summarised across PPO seeds with a t-interval
(the PPO seed is the independent unit).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--baselines", default=str(ROOT / "results" / "evaluation_baselines_90000" / "episodes.csv"))
ap.add_argument("--setup", default="final")
ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 9)))
ap.add_argument("--results-dir", default=str(ROOT / "results"))
ap.add_argument("--sat-threshold", type=float, default=0.15)
args = ap.parse_args()

KEY = ["condition", "seed"]
base = pd.read_csv(args.baselines)
for need in ("ECT", "Greedy"):
    if need not in set(base.agent):
        raise SystemExit(f"{need} rows missing in {args.baselines}. Did you run the baselines-only step after patching?")
ect = base[base.agent == "ECT"].set_index(KEY)["episode_return"]
gre = base[base.agent == "Greedy"].set_index(KEY)["episode_return"]
gre_c = base[base.agent == "Greedy"].groupby("condition")["completion_rate"].mean()
saturated = set(gre_c[gre_c < args.sat_threshold].index)
seen_map = base.drop_duplicates("condition").set_index("condition")["seen"].astype(bool)


def macro(series):
    """mean over conditions (equal weight) of the per-condition mean"""
    return series.groupby(level="condition").mean().mean()


rows, per_cond = [], []
for s in args.seeds:
    path = Path(args.results_dir) / f"evaluation_{args.setup}_s{s}" / "episodes.csv"
    if not path.exists():
        print(f"(missing {path})")
        continue
    d = pd.read_csv(path)
    ppo = d[d.agent == "PPO"].set_index(KEY)["episode_return"]
    g_here = d[d.agent == "Greedy"].set_index(KEY)["episode_return"]
    common = g_here.index.intersection(gre.index)
    if len(common) != len(g_here) or not np.allclose(g_here.loc[common], gre.loc[common], atol=1e-6):
        raise SystemExit(f"Greedy rows differ between {path} and the baseline file: the episodes are not "
                         f"identical, so PPO and ECT cannot be paired. Re-run the baselines with the same "
                         f"--seed-base and --episodes-per-condition.")
    idx = ppo.index.intersection(ect.index)
    p, e, g = ppo.loc[idx], ect.loc[idx], gre.loc[idx]
    cond = idx.get_level_values("condition")
    sat_mask = cond.isin(saturated)
    row = {
        "seed": s, "n_ep": len(idx),
        "ppo_minus_greedy": macro(p - g), "ppo_minus_ect": macro(p - e),
        "ect_minus_greedy": macro(e - g),
        "ppo_minus_ect_headroom": macro((p - e)[~sat_mask]), "ppo_minus_ect_saturated": macro((p - e)[sat_mask]),
        "win_rate_vs_ect": float((p > e).mean()),
    }
    rows.append(row)
    pc = (p - e).groupby(level="condition").mean().rename(f"s{s}")
    per_cond.append(pc)

if not rows:
    raise SystemExit("No PPO result folders found.")
r = pd.DataFrame(rows)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
print("Gaps are PPO minus X, mean over conditions (equal weight); positive = PPO better.\n")
print(r.to_string(index=False))

from scipy import stats
print("\nAcross PPO seeds (seed = independent unit):\n")
for col, label in (("ppo_minus_greedy", "PPO - Greedy"), ("ppo_minus_ect", "PPO - ECT"),
                   ("ppo_minus_ect_headroom", "PPO - ECT (headroom)"), ("ppo_minus_ect_saturated", "PPO - ECT (saturated)")):
    v = r[col].to_numpy()
    if len(v) > 1:
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        h = stats.t.ppf(0.975, len(v) - 1) * se
        print(f"  {label:24s} mean {m:6.2f}  95% CI [{m-h:6.2f}, {m+h:6.2f}]  seeds above 0: {(v > 0).sum()}/{len(v)}")
    else:
        print(f"  {label:24s} {v[0]:6.2f}  (single seed)")
print(f"  ECT - Greedy (same for every seed) {r['ect_minus_greedy'].iloc[0]:6.2f}")

pcdf = pd.concat(per_cond, axis=1)
pcdf.insert(0, "regime", ["saturated" if c in saturated else "headroom" for c in pcdf.index])
pcdf["mean"] = pcdf[[c for c in pcdf.columns if c.startswith("s")]].mean(axis=1)
print("\nPer condition, PPO minus ECT (mean over episodes; columns = PPO seeds):\n")
print(pcdf.sort_values(["regime", "mean"]).to_string())