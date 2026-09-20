"""
experiments/make_figures.py

Figures + a table from one or more evaluation folders (e.g. one per training seed).

    python -m experiments.make_figures "results/evaluation_final_s*"
    python -m experiments.make_figures "results/evaluation_focused_s*" --out results/figures

Writes to --out (default results/figures):
    fig1_forest_vs_greedy.png     PPO - Greedy per run (95% CI), seen vs unseen
    fig2_delta_by_condition.png   PPO - Greedy per condition (bar = mean over runs, dots = runs)
    fig3_completion_by_condition.png  completion rate, Greedy vs PPO (mean, min-max over runs)
    per_condition_delta.csv       the numbers behind fig2
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
ap.add_argument("--baseline", default="Greedy")
args = ap.parse_args()

dirs = []
for pat in args.dirs:
    dirs += sorted(glob.glob(pat)) or [pat]
dirs = [d for d in dirs if (Path(d) / "episodes.csv").exists()]
if not dirs:
    raise SystemExit("No evaluation folders with episodes.csv found.")
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
B = args.baseline

frames, paired = [], []
for d in dirs:
    run = Path(d).name.replace("evaluation_", "")
    e = pd.read_csv(Path(d) / "episodes.csv"); e["run"] = run; frames.append(e)
    pf = Path(d) / "paired_vs_baselines.csv"
    if pf.exists():
        p = pd.read_csv(pf); p = p[p["baseline"] == B].copy(); p["run"] = run; paired.append(p)
df = pd.concat(frames)
runs = sorted(df["run"].unique())

# ---- per-condition table ----
cond = (df[df["agent"].isin(["PPO", B])]
        .groupby(["run", "condition", "seen", "agent"])[["episode_return", "completion_rate"]]
        .mean().unstack("agent"))
tab = pd.DataFrame({
    "delta": cond[("episode_return", "PPO")] - cond[("episode_return", B)],
    "ppo_completion": cond[("completion_rate", "PPO")],
    "base_completion": cond[("completion_rate", B)],
}).reset_index()
summary = tab.groupby(["condition", "seen"]).agg(
    mean_delta=("delta", "mean"), min_delta=("delta", "min"), max_delta=("delta", "max"),
    ppo_completion=("ppo_completion", "mean"), ppo_comp_min=("ppo_completion", "min"),
    ppo_comp_max=("ppo_completion", "max"), base_completion=("base_completion", "mean"),
).reset_index().sort_values("mean_delta")
summary.to_csv(out / "per_condition_delta.csv", index=False)

GOOD, BAD, NEUT = "#2a9d8f", "#e76f51", "#8d99ae"

# ---- fig 1: forest plot ----
if paired:
    P = pd.concat(paired)
    rows = []
    for label in ("seen", "unseen"):
        for r in runs:
            x = P[(P["run"] == r) & (P["conditions"] == label)]
            if len(x):
                x = x.iloc[0]; rows.append((f"{r} [{label}]", x["mean_diff"], x["ci95_low"], x["ci95_high"], label))
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(rows) + 1.6))
    for i, (name, m, lo, hi, label) in enumerate(rows):
        c = GOOD if lo > 0 else BAD if hi < 0 else NEUT
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=c, capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel(f"PPO minus {B}: mean paired episode return (95% bootstrap CI)   → PPO better")
    ax.set_title("Per-run comparison (green = beats, grey = tied, red = loses)")
    fig.tight_layout(); fig.savefig(out / "fig1_forest_vs_greedy.png", dpi=160); plt.close(fig)

# ---- fig 2: delta by condition ----
s = summary.reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8, 0.4 * len(s) + 1.5))
colors = [GOOD if v > 0 else BAD for v in s["mean_delta"]]
ax.barh(range(len(s)), s["mean_delta"], color=colors, alpha=0.75)
for i, (cname, seen) in enumerate(zip(s["condition"], s["seen"])):
    ys = tab[tab["condition"] == cname]["delta"]
    ax.scatter(ys, [i] * len(ys), color="k", s=14, zorder=3)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(range(len(s)))
ax.set_yticklabels([f"{c}{'' if sn else '  (unseen)'}" for c, sn in zip(s["condition"], s["seen"])])
ax.set_xlabel(f"PPO minus {B}: mean episode return   → PPO better   (dots = individual runs)")
ax.set_title(f"Where PPO wins and loses ({len(runs)} run(s))")
fig.tight_layout(); fig.savefig(out / "fig2_delta_by_condition.png", dpi=160); plt.close(fig)

# ---- fig 3: completion rate ----
s3 = summary.sort_values("base_completion", ascending=False).reset_index(drop=True)
x = np.arange(len(s3)); w = 0.4
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x - w / 2, s3["base_completion"] * 100, w, label=B, color=NEUT)
yerr = [(s3["ppo_completion"] - s3["ppo_comp_min"]) * 100, (s3["ppo_comp_max"] - s3["ppo_completion"]) * 100]
ax.bar(x + w / 2, s3["ppo_completion"] * 100, w, yerr=yerr, capsize=2, label="PPO (mean, min–max over runs)", color=GOOD)
ax.set_xticks(x); ax.set_xticklabels(s3["condition"], rotation=60, ha="right")
ax.set_ylabel("Completion rate (%)"); ax.legend(); ax.set_title("Task completion by condition")
fig.tight_layout(); fig.savefig(out / "fig3_completion_by_condition.png", dpi=160); plt.close(fig)

print(f"Read {len(runs)} run(s): {runs}")
print(f"Wrote figures + per_condition_delta.csv to {out}")
print("\nPer-condition PPO - baseline (mean over runs), worst to best:")
print(summary[["condition", "seen", "mean_delta", "min_delta", "max_delta"]].round(1).to_string(index=False))