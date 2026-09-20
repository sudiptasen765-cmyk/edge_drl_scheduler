"""
experiments/explain_policy.py

Explainability for the trained masked-PPO scheduler (no extra dependencies).

  1. ACTION SHARES   which servers the policy sends tasks to, per condition
                     (skew under heavy load = "load concentration" hypothesis).
  2. PERMUTATION IMPORTANCE  how often the policy's chosen server changes when a
                     feature (or a whole feature TYPE across servers) is shuffled
                     across decision states. Reported separately for a "light"
                     regime and a "heavy" regime.

    python -m experiments.explain_policy --model models/ppo_checkpoints/focused_s3/best_model
    python -m experiments.explain_policy --model ... --episodes-per-condition 5 --out results/explain_s3

Caveats: permuting one feature at a time creates states the policy may never see and
correlated features share credit, so treat flip-rates as relative rankings, not exact
causal effects. Only states reached BY THE POLICY are analysed.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from environment.edge_scheduling_env import EdgeSchedulingEnv
from training.train_ppo import training_conditions

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "models" / "ppo_checkpoints" / "best_model"))
ap.add_argument("--episodes-per-condition", type=int, default=3)
ap.add_argument("--seed-base", type=int, default=70_000)
ap.add_argument("--light-workloads", nargs="+", default=["normal", "variable"])
ap.add_argument("--repeats", type=int, default=3, help="permutation repeats per feature")
ap.add_argument("--out", default=str(ROOT / "results" / "explain"))
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

env = EdgeSchedulingEnv.from_config_files()
zip_path = args.model if args.model.endswith(".zip") else args.model + ".zip"
model = MaskablePPO.load(zip_path, device="cpu")
workloads, networks = training_conditions(env)
server_ids = list(env.server_ids)
names = list(env.feature_names)
rng = np.random.default_rng(0)

# ---------------- collect states visited by the policy ----------------
O, M, A, W, C = [], [], [], [], []
for w in workloads:
    for n in networks:
        for i in range(args.episodes_per_condition):
            seed = args.seed_base + i
            obs, _ = env.reset(seed=seed, options={"workload_type": w, "network_scenario": n, "task_seed": seed})
            done = False
            while not done:
                mask = env.action_masks()
                a, _ = model.predict(obs, action_masks=mask, deterministic=True)
                a = int(np.asarray(a).reshape(-1)[0])
                O.append(obs.copy()); M.append(mask.copy()); A.append(a); W.append(w); C.append(f"{w}_{n}")
                obs, _, term, trunc, _ = env.step(a)
                done = term or trunc
O, M, A = np.array(O), np.array(M), np.array(A); W = np.array(W); C = np.array(C)
print(f"Collected {len(O)} decision states from {len(set(C))} conditions.")

# ---------------- 1. action shares ----------------
shares = pd.DataFrame({"condition": C, "server": [server_ids[a] for a in A]})
share_tab = (pd.crosstab(shares["condition"], shares["server"], normalize="index")
             .reindex(columns=server_ids, fill_value=0.0) * 100)   # servers never chosen -> 0%
share_tab["top_share_%"] = share_tab.max(axis=1)
p = share_tab[server_ids].to_numpy() / 100
with np.errstate(divide="ignore", invalid="ignore"):
    share_tab["norm_entropy"] = -(np.where(p > 0, p * np.log(p), 0)).sum(axis=1) / np.log(len(server_ids))
share_tab.round(1).to_csv(out / "action_shares.csv")
print("\n=== Share of decisions sent to each server (%), per condition ===")
print("(norm_entropy: 1.0 = perfectly even across servers, 0 = everything on one server)\n")
print(share_tab.round(2).to_string())

d0 = env.decode_observation(O[0])["servers"]
prof = pd.DataFrame({sid: d0[sid] for sid in server_ids}).T
static = [c for c in ("cpu_cores", "ram_gb", "bandwidth", "power_max") if c in prof.columns]
print("\n=== Static server profile (normalised so the largest fleet value = 1) ===")
print(prof[static].round(2).to_string())


# ---------------- 2. permutation importance ----------------
def feature_type(name: str) -> str:
    toks = name.split("_")
    kept = [t for t in toks if not (t.isdigit() or (t[:1] in "s" and t[1:].isdigit())
                                    or (t.startswith("srv") and t[3:].isdigit())
                                    or (t.startswith("server") and t[6:].isdigit()))]
    ftype = "_".join(kept) if len(kept) < len(toks) else name
    return ("task: " if name.startswith("task") else "server: ") + ftype


types = {}
for j, nme in enumerate(names):
    types.setdefault(feature_type(nme), []).append(j)


def flip_rate(X, mask, base, cols):
    rates = []
    for _ in range(args.repeats):
        Xp = X.copy()
        perm = rng.permutation(len(X))
        Xp[:, cols] = X[perm][:, cols]          # one row-permutation for the whole group
        a, _ = model.predict(Xp, action_masks=mask, deterministic=True)
        rates.append(np.mean(np.asarray(a) != base))
    return float(np.mean(rates))


regimes = {"light": np.isin(W, args.light_workloads), "heavy": ~np.isin(W, args.light_workloads)}
feat_rows, type_rows = [], []
for reg, sel in regimes.items():
    if sel.sum() < 20:
        continue
    X, mk = O[sel], M[sel]
    base, _ = model.predict(X, action_masks=mk, deterministic=True)
    base = np.asarray(base)
    for j, nme in enumerate(names):
        feat_rows.append({"regime": reg, "feature": nme, "flip_rate": flip_rate(X, mk, base, [j])})
    for t, cols in types.items():
        type_rows.append({"regime": reg, "type": t, "n_cols": len(cols), "flip_rate": flip_rate(X, mk, base, cols)})

F = pd.DataFrame(feat_rows); T = pd.DataFrame(type_rows)
F.to_csv(out / "feature_importance.csv", index=False); T.to_csv(out / "type_importance.csv", index=False)
pd.set_option("display.width", 200)
print("\n=== Importance by feature TYPE (fraction of decisions that change when the type is shuffled) ===\n")
tt = T.pivot(index="type", columns="regime", values="flip_rate")
print((tt.sort_values(tt.columns[0], ascending=False) * 100).round(1).to_string())
for reg in F["regime"].unique():
    print(f"\n=== Top 12 single features, {reg} regime ===")
    print(F[F["regime"] == reg].sort_values("flip_rate", ascending=False).head(12)
          .assign(flip_pct=lambda d: (d["flip_rate"] * 100).round(1))[["feature", "flip_pct"]].to_string(index=False))

# ---------------- figure ----------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tt2 = tt.loc[tt.max(axis=1).sort_values().index] * 100
    ax = tt2.plot.barh(figsize=(8, 0.35 * len(tt2) + 1.5), color=["#2a9d8f", "#e76f51"][: tt2.shape[1]])
    ax.set_xlabel("% of decisions that change when this feature type is shuffled")
    ax.set_title("What drives the PPO policy's server choice")
    plt.tight_layout(); plt.savefig(out / "feature_type_importance.png", dpi=160)
except Exception as e:  # plotting is optional
    print(f"(figure skipped: {e})")
print(f"\nWrote CSVs (+ figure) to {out}")