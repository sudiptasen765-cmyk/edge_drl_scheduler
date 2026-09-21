"""
explainability/explain_decision.py - "why did the scheduler pick that server?"

For one decision (observation + action mask) of a trained MaskablePPO policy
this returns:
  * the chosen server and the MASKED action probabilities (illegal servers = 0);
  * which servers were eligible;
  * the state features that pushed the policy towards its choice (and against it);
  * the same information grouped by block (task / each server);
  * a short human-readable sentence.

Method (occlusion / sensitivity analysis - model-based, NOT a causal claim)
--------------------------------------------------------------------------
margin(x)    = log p(chosen | x) - log p(runner-up | x)   over ELIGIBLE actions
                (== logit difference, because masking renormalises equally)
effect_i     = margin(x) - margin(x with feature i replaced by its reference value)
The reference value of a feature is its mean over states the policy itself visits
(`build_reference`). effect_i > 0 means "the current value of feature i makes the
policy prefer the chosen server over the runner-up, compared with a typical
state"; < 0 means it argues against.

Feature FAMILIES
----------------
Single-feature replacement can create an inconsistent state (e.g. one server looks
half full while every other queue looks empty), and the network's answer to such a
state says little. So the same analysis is repeated for FAMILIES of features that
move together: each per-server quantity across all servers (queue_fill, backlog,
net_delay, can_start_now, ...), task priority (the three one-hot flags), and each
remaining task feature. Prefer the family view for reading a decision.

Caveats, deliberately stated:
  * Features are replaced one at a time. Interactions between features (e.g.
    "backlog matters only if slack is small") are not captured by a single
    number; the effects do not add up to the margin.
  * The action mask is held fixed while a feature is replaced (it is a function
    of the true state, not of the network input).
  * Observation values are the environment's normalised features in [0, 1]
    (or [-1, 1]); they are not raw milliseconds or percentages.

    python -m explainability.explain_decision \
        --model models/ppo_edge_scheduler_selected.zip --workload normal --network normal --skip 40 --steps 5
"""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

_SERVER_RE = re.compile(r"^server(\d+)_(.+)$")


@dataclass
class FeatureEffect:
    name: str
    value: float          # current (normalised) value
    reference: float      # typical value under the policy's own states
    effect: float         # >0 supports the chosen server, <0 argues against
    group: str            # "task" or "server<id>"


@dataclass
class Explanation:
    chosen_server: int
    runner_up_server: int | None
    probabilities: dict[int, float]          # server id -> masked probability
    eligible_servers: list[int]
    margin: float | None                     # log-prob gap chosen vs runner-up
    supporting: list[FeatureEffect] = field(default_factory=list)
    opposing: list[FeatureEffect] = field(default_factory=list)
    group_effects: dict[str, float] = field(default_factory=dict)
    # single features whose replacement by the typical value would CHANGE the chosen
    # server: [{"name", "becomes_server", "effect"}], largest effect first. Empty = the
    # decision is robust to any one feature.
    flip_features: list[dict] = field(default_factory=list)
    # same analysis for FAMILIES of features replaced together:
    # [{"family", "effect", "n_features", "becomes_server" (None if the choice stays)}]
    family_effects: list[dict] = field(default_factory=list)
    text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------
# model access
# ----------------------------------------------------------------------

def _masked_probs(model, obs_batch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(N, n_actions) probabilities with illegal actions forced to 0."""
    x = torch.as_tensor(np.asarray(obs_batch, dtype=np.float32), device=model.device)
    masks = np.tile(np.asarray(mask, dtype=bool), (len(x), 1))
    with torch.no_grad():
        dist = model.policy.get_distribution(x, action_masks=masks)
        return dist.distribution.probs.cpu().numpy()


def build_reference(model, env, n_episodes: int = 8, seed: int = 0) -> np.ndarray:
    """Mean observation over states the trained policy actually visits."""
    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        while True:
            rows.append(np.asarray(obs, dtype=np.float32))
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
    return np.mean(rows, axis=0)


# ----------------------------------------------------------------------
# core
# ----------------------------------------------------------------------

def feature_families(feature_names: list[str]) -> dict[str, list[int]]:
    """Family name -> feature indices. Per-server quantities are grouped ACROSS servers."""
    fam: dict[str, list[int]] = {}
    for i, name in enumerate(feature_names):
        m = _SERVER_RE.match(name)
        if m:
            key = f"{m.group(2)} (all servers)"
        elif name.startswith("task_prio_"):
            key = "task priority"
        else:
            key = name
        fam.setdefault(key, []).append(i)
    return fam


def _group_of(name: str) -> str:
    m = _SERVER_RE.match(name)
    return f"server{m.group(1)}" if m else "task"


def explain_decision(
    model,
    obs: np.ndarray,
    mask: np.ndarray,
    feature_names: list[str],
    server_ids: list[int],
    reference: np.ndarray,
    top_k: int = 3,
) -> Explanation:
    obs = np.asarray(obs, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    reference = np.asarray(reference, dtype=np.float32)
    if len(feature_names) != obs.shape[0] or reference.shape != obs.shape:
        raise ValueError("feature_names, obs and reference must have the same length.")
    if len(server_ids) != mask.shape[0]:
        raise ValueError("server_ids and mask must have the same length.")

    probs = _masked_probs(model, obs[None], mask)[0]
    chosen = int(np.argmax(probs))
    eligible_idx = [i for i in range(len(mask)) if mask[i]]
    exp = Explanation(
        chosen_server=server_ids[chosen],
        runner_up_server=None,
        probabilities={sid: float(p) for sid, p in zip(server_ids, probs)},
        eligible_servers=[server_ids[i] for i in eligible_idx],
        margin=None,
    )

    others = [i for i in eligible_idx if i != chosen]
    if not others:
        exp.text = (f"Server {exp.chosen_server} was the only eligible server "
                    f"(all others are masked), so no comparison is possible.")
        return exp

    runner = max(others, key=lambda i: probs[i])
    exp.runner_up_server = server_ids[runner]

    def margin(p: np.ndarray) -> np.ndarray:
        eps = 1e-12
        return np.log(p[:, chosen] + eps) - np.log(p[:, runner] + eps)

    base_margin = float(margin(probs[None])[0])
    exp.margin = base_margin

    # occlusion: replace one feature at a time by its reference value (one batched pass)
    n = obs.shape[0]
    perturbed = np.tile(obs, (n, 1))
    perturbed[np.arange(n), np.arange(n)] = reference
    perturbed_probs = _masked_probs(model, perturbed, mask)
    effects = base_margin - margin(perturbed_probs)
    new_choice = perturbed_probs.argmax(axis=1)

    feats = [
        FeatureEffect(name=feature_names[i], value=float(obs[i]), reference=float(reference[i]),
                      effect=float(effects[i]), group=_group_of(feature_names[i]))
        for i in range(n)
    ]
    exp.flip_features = sorted(
        ({"name": feature_names[i], "becomes_server": server_ids[int(new_choice[i])], "effect": float(effects[i])}
         for i in range(n) if int(new_choice[i]) != chosen),
        key=lambda d: -abs(d["effect"]),
    )[:top_k]
    exp.supporting = sorted((f for f in feats if f.effect > 0), key=lambda f: -f.effect)[:top_k]
    exp.opposing = sorted((f for f in feats if f.effect < 0), key=lambda f: f.effect)[:top_k]
    groups: dict[str, float] = {}
    for f in feats:
        groups[f.group] = groups.get(f.group, 0.0) + f.effect
    exp.group_effects = dict(sorted(groups.items(), key=lambda kv: -abs(kv[1])))

    fams = feature_families(feature_names)
    fam_obs = np.tile(obs, (len(fams), 1))
    for row, idx in enumerate(fams.values()):
        fam_obs[row, idx] = reference[idx]
    fam_probs = _masked_probs(model, fam_obs, mask)
    fam_eff = base_margin - margin(fam_probs)
    fam_choice = fam_probs.argmax(axis=1)
    exp.family_effects = sorted(
        ({"family": name, "effect": float(fam_eff[r]), "n_features": len(idx),
          "becomes_server": (server_ids[int(fam_choice[r])] if int(fam_choice[r]) != chosen else None)}
         for r, (name, idx) in enumerate(fams.items())),
        key=lambda d: -abs(d["effect"]),
    )
    exp.text = _sentence(exp, top_k)
    return exp


def _sentence(e: Explanation, top_k: int = 3) -> str:
    pc, pr = e.probabilities[e.chosen_server], e.probabilities[e.runner_up_server]
    s = (f"Chose server {e.chosen_server} (p={pc:.2f}) over server {e.runner_up_server} "
         f"(p={pr:.2f}, log-odds margin {e.margin:.1f}); eligible: {e.eligible_servers}.")
    sup = [f for f in e.family_effects if f["effect"] > 0][:top_k]
    opp = [f for f in e.family_effects if f["effect"] < 0][:1]
    if sup:
        s += " Factor groups favouring this choice: " + "; ".join(
            f"{f['family']} ({f['effect']:+.1f})" for f in sup) + "."
    if opp:
        s += f" Strongest group against: {opp[0]['family']} ({opp[0]['effect']:+.1f})."
    flips = [f for f in e.family_effects if f["becomes_server"] is not None]
    if flips:
        f = flips[0]
        s += (f" Resetting {f['family']} to its typical value would switch the choice to "
              f"server {f['becomes_server']}.")
    else:
        s += " No single factor group reset to its typical value would change this choice."
    s += " (Sensitivity analysis of the model, not a causal explanation.)"
    return s


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv=None):
    from sb3_contrib import MaskablePPO

    from environment.edge_scheduling_env import EdgeSchedulingEnv

    ap = argparse.ArgumentParser(description="Explain the first decisions of an episode.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", default="normal")
    ap.add_argument("--network", default="normal")
    ap.add_argument("--task-seed", type=int, default=12345)
    ap.add_argument("--steps", type=int, default=5, help="number of decisions to explain")
    ap.add_argument("--skip", type=int, default=0,
                    help="follow the model silently for this many decisions first "
                         "(early decisions see empty queues and are not representative)")
    ap.add_argument("--reference-episodes", type=int, default=8)
    args = ap.parse_args(argv)

    model = MaskablePPO.load(args.model, device="cpu")
    env = EdgeSchedulingEnv.from_config_files()
    names = list(getattr(env, "feature_names", None) or [f"f{i}" for i in range(env.observation_space.shape[0])])
    sids = list(getattr(env, "server_ids", None) or range(env.action_space.n))
    print(f"Building reference from {args.reference_episodes} policy episodes...")
    ref = build_reference(model, env, args.reference_episodes)

    obs, info = env.reset(options={"workload_type": args.workload, "network_scenario": args.network,
                                   "task_seed": args.task_seed})
    done = False
    for _ in range(args.skip):
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            done = True
            break
    if done:
        raise SystemExit(f"The episode ended before decision {args.skip}; use a smaller --skip.")
    for step in range(args.steps):
        mask = env.action_masks()
        e = explain_decision(model, obs, mask, names, sids, ref)
        print(f"\n[decision {args.skip + step + 1}] {e.text}")
        top = list(e.group_effects.items())[:3]
        print("  block effects: " + ", ".join(f"{g} {v:+.2f}" for g, v in top))
        print("  top single features (treat with care - see docstring): " + ", ".join(
            f"{f.name}={f.value:.2f} ({f.effect:+.1f})" for f in (e.supporting[:2] + e.opposing[:1])))
        obs, _, terminated, truncated, _ = env.step(sids.index(e.chosen_server))
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()