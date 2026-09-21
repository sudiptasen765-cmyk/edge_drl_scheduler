"""
training/train_ppo_bc.py

Warm-starts MaskablePPO by behaviour-cloning the ECT scheduler (policy AND
value networks), then fine-tunes with PPO at a low learning rate.

Usage:
    python -m training.train_ppo_bc --run-name bc_s1 --seed 1
    python -m training.train_ppo_bc --run-name bc_s1 --seed 1 --timesteps 2000   (smoke test)
    python -m training.train_ppo_bc --run-name bc_s1 --seed 1 --finetune-lr 5e-5

Then, matching the existing workflow:
    python -m experiments.evaluate_agents --include-unseen \
        --model models/ppo_checkpoints/bc_s1/best_model --seed-base 90000 \
        --episodes-per-condition 20 --out results/evaluation_bc_s1
    python -m experiments.compare_ppo_ect --setup bc --seeds 1 2 3 4 5

Why ECT, and why both networks
-------------------------------
ECT (scheduling/ect_scheduler.py) beats PPO specifically on the "headroom"
conditions (normal_normal, variable_normal, normal_unseen - see
results/selected_model.md), by reasoning about per-server completion time
(network delay + backlog) that Greedy ignores and PPO has apparently not
learned cleanly from reward alone. Cloning ECT's DECISIONS gives the policy
head a head start on exactly that skill before any PPO gradient is taken.
Cloning the VALUE head too (regressing towards ECT's discounted
return-to-go, on rewards from the SAME distribution PPO will fine-tune on)
avoids the value function starting from scratch and fighting the policy's
advantage estimates for the first N updates - which is what killed
explained_variance in the very first (non-warm-started, non-normalized)
training run this project did.

Two-stage procedure
--------------------
1. COLLECT: roll ECT through ONE training-style environment (same wrapper
   stack train_ppo.py builds for real training: Monitor -> DummyVecEnv,
   optionally VecNormalize) for --bc-episodes episodes, recording every
   (observation, action mask, chosen action, reward). Held-out task seeds,
   disjoint from every other seed range in this project (see
   BC_DATA_SEED_BASE below).
2. CLONE: build the real MaskablePPO model (this creates model.policy with
   fresh random weights and sets up its optimizer at --finetune-lr). Before
   calling .learn(), run supervised epochs directly on model.policy:
     - policy loss  = mean negative log-likelihood of ECT's action under the
                      CURRENT policy, with the SAME action masking PPO uses
                      at inference (model.policy.evaluate_actions(...,
                      action_masks=...) - verified against sb3_contrib
                      2.8.0's actual source, see the module-level note below).
     - value loss   = MSE between the value head's prediction and ECT's
                      discounted return-to-go.
   If reward normalization is on, the fine-tuning VecNormalize's ret_rms
   (running reward-variance statistics) is seeded from the SAME statistics
   the BC dataset's returns were computed under, using sb3's own
   RunningMeanStd instance (verified attribute name against source) - so
   the value head's regression target scale during BC pretraining matches
   what it will keep being trained against once PPO fine-tuning starts, and
   we don't get a scale-shock at the moment .learn() begins.
   Then model.learn(total_timesteps=..., ...) runs ordinary PPO fine-tuning
   on top of the warm-started weights, with the SAME BaselineEvalCallback
   (vs Greedy) and checkpoint pattern as train_ppo.py, so the run integrates
   with the existing evaluate_agents.py / compare_ppo_ect.py pipeline
   unchanged.

Correctness notes (verified against the pinned sb3-contrib==2.8.0 /
stable-baselines3==2.8.0 source, since this project pins those versions and
an untested guess at internal APIs would be worse than not writing this):
  - MaskableActorCriticPolicy.evaluate_actions(obs, actions, action_masks)
    returns (values, log_prob, entropy); log_prob is the log-likelihood of
    `actions` under the CURRENT policy with masking already applied, i.e.
    -log_prob.mean() IS the behaviour-cloning cross-entropy loss.
  - action_masks may be passed as a plain numpy bool array; the distribution
    converts it internally (th.as_tensor(masks, dtype=th.bool)).
  - VecEnv.seed(seed) stores seed+idx per sub-env, consumed and cleared by
    the NEXT reset() call (not applied instantly) - so the call order below
    (seed() then reset()) is required, not just convention.
  - get_action_masks(vec_env) (sb3_contrib.common.maskable.utils) is
    np.stack(vec_env.env_method("action_masks")) - used directly here rather
    than reimplemented, so BC data collection asks for masks exactly the way
    MaskablePPO itself does during real training.
  - VecNormalize keeps its running reward-variance statistics in
    `.ret_rms` (a stable_baselines3.common.running_mean_std.RunningMeanStd),
    updated as `self.returns = self.returns * gamma + reward;
    self.ret_rms.update(self.returns)` - matching this script's own
    return-to-go computation below, so copying `.ret_rms` across is a valid
    way to carry the statistics forward.

What I could NOT test
----------------------
The actual DummyVecEnv + VecNormalize + MaskablePPO integration below could
not be run in my environment (no working torch/sb3-contrib install
available to me). What WAS tested there: the ECT rollout -> action-index
mapping -> discounted-return-to-go pipeline, directly against the real
EdgeSchedulingEnv and EarliestCompletionScheduler (368 decisions across 5
episodes, zero masked-action violations, correct return-to-go values).
Please run the --timesteps 2000 smoke test below before any real run and
paste the output - the same way every other phase of this project has been
verified.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from agents.ppo_config import PPOConfig
from scheduling.ect_scheduler import EarliestCompletionScheduler
from scheduling.greedy_scheduler import GreedyScheduler
from training.train_ppo import (
    DEFAULT_PPO_CONFIG,
    PROJECT_ROOT,
    BaselineEvalCallback,
    build_eval_suite,
    build_vec_env,
    set_global_seed,
    training_conditions,
)
from environment.edge_scheduling_env import EdgeSchedulingEnv

# Distinct from: training seeds (small ints), the mid-training eval callback
# (10_000+), and experiments/evaluate_agents.py's held-out seeds (50_000+,
# 90_000+). Nothing here should ever face the same task stream as evaluation.
BC_DATA_SEED_BASE = 500_000


# ----------------------------------------------------------------------
# stage 1: collect a behaviour-cloning dataset from ECT
# ----------------------------------------------------------------------

def collect_bc_dataset(
    cfg: PPOConfig,
    env_overrides: dict | None,
    n_episodes: int,
    use_reward_norm: bool,
    seed_base: int = BC_DATA_SEED_BASE,
) -> tuple[dict[str, np.ndarray], "VecNormalize | None"]:
    """
    Roll EarliestCompletionScheduler through ONE training-style environment
    (same wrapper stack build_vec_env() gives real training, num_envs=1) for
    n_episodes, recording every decision.

    Returns:
      dataset: {"obs", "masks", "actions", "returns", "episode_id"} arrays,
               one row per decision, concatenated across all episodes.
      bc_vec_env: the VecNormalize instance used to collect this data (None
               if use_reward_norm is False) - kept alive so its `.ret_rms`
               can be transferred to the fine-tuning vec_env afterwards.
    """
    scheduler = EarliestCompletionScheduler()
    vec_env = build_vec_env(1, env_overrides)
    if use_reward_norm:
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=cfg.ppo.gamma)

    obs_list, mask_list, action_list, reward_list, episode_id_list = [], [], [], [], []
    ep_boundaries = [0]

    for ep in range(n_episodes):
        vec_env.seed(seed_base + ep)  # consumed by the reset() call right below
        obs = vec_env.reset()
        done = False
        while not done:
            mask = get_action_masks(vec_env)[0]
            sim = vec_env.get_attr("sim")[0]
            server_ids = vec_env.get_attr("server_ids")[0]
            task = sim.next_pending_task()
            server_id = scheduler(task, sim)
            action = server_ids.index(server_id)
            if not mask[action]:
                raise RuntimeError(
                    f"ECT chose action {action} (server {server_id}) but the action "
                    f"mask marks it invalid: {mask}. This would mean ECT and the "
                    "environment's own feasibility check disagree - stopping rather "
                    "than silently training on a bad example."
                )

            obs_list.append(np.asarray(obs)[0].copy())
            mask_list.append(mask.copy())
            action_list.append(action)
            episode_id_list.append(ep)

            obs, reward, dones, _infos = vec_env.step(np.array([action]))
            # ORIGINAL (un-normalised) reward: VecNormalize's divisor drifts during collection
            reward_list.append(float(vec_env.get_original_reward()[0]) if use_reward_norm else float(reward[0]))
            done = bool(dones[0])
        ep_boundaries.append(len(obs_list))

    # Discounted return-to-go per decision, matching VecNormalize's own
    # update rule (self.returns = self.returns * gamma + reward) so the
    # value-regression target is on the same footing the value head will
    # keep being trained against once PPO fine-tuning takes over.
    returns = np.zeros(len(reward_list), dtype=np.float32)
    for i in range(len(ep_boundaries) - 1):
        lo, hi = ep_boundaries[i], ep_boundaries[i + 1]
        running = 0.0
        for t in range(hi - 1, lo - 1, -1):
            running = reward_list[t] + cfg.ppo.gamma * running
            returns[t] = running

    if use_reward_norm:
        # one common scale: the final running std, which is what fine-tuning starts from
        returns = (returns / np.sqrt(vec_env.ret_rms.var + vec_env.epsilon)).astype(np.float32)

    dataset = {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "masks": np.asarray(mask_list, dtype=bool),
        "actions": np.asarray(action_list, dtype=np.int64),
        "returns": returns,
        "episode_id": np.asarray(episode_id_list, dtype=np.int64),
    }
    return dataset, (vec_env if use_reward_norm else None)


# ----------------------------------------------------------------------
# stage 2: behaviour cloning on model.policy
# ----------------------------------------------------------------------

def behavior_clone(
    model: MaskablePPO,
    dataset: dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
    lr: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    val_fraction: float,
    seed: int,
) -> None:
    """Supervised warm-start of model.policy (both the policy and value
    heads) on the ECT dataset. Uses its OWN optimizer, separate from the
    PPO optimizer that .learn() will use afterwards (model.policy.optimizer
    is left untouched here, still at its --finetune-lr construction value)."""
    n = len(dataset["actions"])
    episodes = np.unique(dataset["episode_id"])
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_val_eps = max(1, int(len(episodes) * val_fraction))
    val_eps = set(episodes[:n_val_eps].tolist())
    val_mask = np.isin(dataset["episode_id"], list(val_eps))
    train_idx = np.nonzero(~val_mask)[0]
    val_idx = np.nonzero(val_mask)[0]
    print(f"BC dataset: {n} decisions across {len(episodes)} episodes "
          f"({len(train_idx)} train / {len(val_idx)} val, split by episode)")

    policy = model.policy
    policy.set_training_mode(True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    def run_epoch(idx: np.ndarray, train: bool) -> dict:
        if train:
            rng.shuffle(idx)
        total_policy_loss = total_value_loss = total_entropy = total_correct = 0.0
        n_seen = 0
        for start in range(0, len(idx), batch_size):
            batch = idx[start:start + batch_size]
            obs_tensor, _ = policy.obs_to_tensor(dataset["obs"][batch])
            actions_tensor = torch.as_tensor(dataset["actions"][batch], device=model.device)
            returns_tensor = torch.as_tensor(dataset["returns"][batch], device=model.device)
            masks_batch = dataset["masks"][batch]

            values, log_prob, entropy = policy.evaluate_actions(
                obs_tensor, actions_tensor, action_masks=masks_batch
            )
            policy_loss = -log_prob.mean()
            value_loss = F.mse_loss(values.flatten(), returns_tensor)
            entropy_loss = -entropy.mean()
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

            with torch.no_grad():
                masked_logits = policy.get_distribution(obs_tensor, action_masks=masks_batch).distribution.logits
                pred = masked_logits.argmax(dim=-1)
                total_correct += (pred == actions_tensor).sum().item()
            total_policy_loss += policy_loss.item() * len(batch)
            total_value_loss += value_loss.item() * len(batch)
            total_entropy += entropy.mean().item() * len(batch)
            n_seen += len(batch)

        return {
            "policy_loss": total_policy_loss / n_seen,
            "value_loss": total_value_loss / n_seen,
            "entropy": total_entropy / n_seen,
            "accuracy": total_correct / n_seen,
        }

    for epoch in range(1, epochs + 1):
        tr = run_epoch(train_idx.copy(), train=True)
        policy.set_training_mode(False)
        with torch.no_grad():
            va = run_epoch(val_idx.copy(), train=False)
        policy.set_training_mode(True)
        print(f"  BC epoch {epoch:>2}/{epochs}  "
              f"train: loss={tr['policy_loss']:.4f} vloss={tr['value_loss']:.4f} "
              f"acc={tr['accuracy']*100:.1f}%  |  "
              f"val: loss={va['policy_loss']:.4f} vloss={va['value_loss']:.4f} "
              f"acc={va['accuracy']*100:.1f}%")

    policy.set_training_mode(False)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Warm-start MaskablePPO via behaviour cloning on ECT, then PPO fine-tune."
    )
    ap.add_argument("--config", type=str, default=str(DEFAULT_PPO_CONFIG))
    ap.add_argument("--timesteps", type=int, default=None,
                     help="Override training.total_timesteps for the PPO fine-tune stage "
                          "(e.g. --timesteps 2000 for a quick smoke test).")
    ap.add_argument("--run-name", type=str, required=True,
                     help="Checkpoints under models/ppo_checkpoints/<run-name>/, "
                          "TensorBoard as maskable_ppo_<run-name>.")
    ap.add_argument("--seed", type=int, default=None, help="Override training.seed.")
    ap.add_argument("--train-workloads", nargs="+", default=["normal", "variable"],
                     help="Training AND BC-collection distribution. Default matches the "
                          "'focused' setup (results/selected_model.md), the strongest known "
                          "PPO configuration so far. Evaluation always uses the full config.")
    ap.add_argument("--train-networks", nargs="+", default=None)
    ap.add_argument("--no-reward-norm", action="store_true")
    ap.add_argument("--eval-per-condition", type=int, default=None)
    # --- BC-specific ---
    ap.add_argument("--bc-episodes", type=int, default=300,
                     help="ECT episodes to collect for behaviour cloning.")
    ap.add_argument("--bc-epochs", type=int, default=20)
    ap.add_argument("--bc-batch-size", type=int, default=256)
    ap.add_argument("--bc-lr", type=float, default=1e-3,
                     help="Adam learning rate for the supervised BC stage (separate from "
                          "--finetune-lr, which governs the PPO stage afterwards).")
    ap.add_argument("--bc-vf-coef", type=float, default=0.5)
    ap.add_argument("--bc-ent-coef", type=float, default=0.01,
                     help="Small entropy bonus during BC so the cloned policy isn't fully "
                          "deterministic going into PPO fine-tuning.")
    ap.add_argument("--bc-val-fraction", type=float, default=0.1)
    ap.add_argument("--finetune-lr", type=float, default=1e-4,
                     help="PPO learning rate for the fine-tuning stage. Deliberately lower "
                          "than config/ppo_config.yaml's from-scratch default (3e-4) so "
                          "fine-tuning nudges the cloned policy rather than overwriting it.")
    args = ap.parse_args()

    cfg = PPOConfig.from_yaml(args.config)
    total_timesteps = args.timesteps if args.timesteps is not None else cfg.training.total_timesteps
    buffer_size = cfg.ppo.n_steps * cfg.training.num_envs
    if buffer_size % cfg.ppo.batch_size != 0:
        raise ValueError(
            f"ppo.n_steps ({cfg.ppo.n_steps}) * training.num_envs "
            f"({cfg.training.num_envs}) = {buffer_size} is not a multiple of "
            f"ppo.batch_size ({cfg.ppo.batch_size})."
        )

    seed = args.seed if args.seed is not None else cfg.training.seed
    set_global_seed(seed)

    checkpoint_dir = PROJECT_ROOT / cfg.training.checkpoint_dir / args.run_name
    tensorboard_dir = PROJECT_ROOT / cfg.training.tensorboard_log
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    env_section = {"workloads": list(args.train_workloads)}
    if args.train_networks:
        env_section["network_scenarios"] = list(args.train_networks)
    train_overrides = {"environment": env_section}
    print(f"Training and BC-collection restricted to: {env_section} "
          "(evaluation still uses ALL conditions)")

    use_reward_norm = not args.no_reward_norm

    # ---- stage 1: collect ECT demonstrations ----
    print(f"\nCollecting {args.bc_episodes} ECT episodes for behaviour cloning "
          f"(seeds {BC_DATA_SEED_BASE + 10_000 * seed}+)...")
    t0 = time.time()
    dataset, bc_vec_env = collect_bc_dataset(
        cfg, train_overrides, args.bc_episodes, use_reward_norm,
        seed_base=BC_DATA_SEED_BASE + 10_000 * seed
    )
    print(f"  collected {len(dataset['actions'])} decisions in {time.time() - t0:.0f}s")

    # ---- build the real fine-tuning vec_env ----
    finetune_vec_env = build_vec_env(cfg.training.num_envs, train_overrides)
    finetune_vec_env.seed(seed)
    if use_reward_norm:
        finetune_vec_env = VecNormalize(
            finetune_vec_env, norm_obs=False, norm_reward=True, gamma=cfg.ppo.gamma
        )
        # Carry the reward-variance statistics forward from BC collection, so
        # the value head's regression target scale doesn't jump the moment
        # PPO fine-tuning starts stepping a *different* VecNormalize instance.
        before = (float(bc_vec_env.ret_rms.mean), float(bc_vec_env.ret_rms.var), int(bc_vec_env.ret_rms.count))
        finetune_vec_env.ret_rms = bc_vec_env.ret_rms
        print(f"Reward normalization: ON - transferred ret_rms from BC collection "
              f"(mean={before[0]:.3f}, var={before[1]:.3f}, count={before[2]})")
    else:
        print("Reward normalization: OFF")

    eval_env = EdgeSchedulingEnv.from_config_files()
    workloads, networks = training_conditions(eval_env)
    n_conditions = len(workloads) * len(networks)
    import math
    per_condition = args.eval_per_condition or max(1, math.ceil(cfg.training.eval_episodes / n_conditions))
    eval_suite = build_eval_suite(workloads, networks, per_condition)
    print(f"Eval suite: {len(eval_suite)} episodes (condition-balanced, fixed seeds)")

    print(f"\nBuilding MaskablePPO model (finetune_lr={args.finetune_lr})...")
    model = MaskablePPO(
        policy=cfg.ppo.policy,
        env=finetune_vec_env,
        learning_rate=args.finetune_lr,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_range=cfg.ppo.clip_range,
        ent_coef=cfg.ppo.ent_coef,
        vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        target_kl=cfg.ppo.target_kl,
        normalize_advantage=cfg.ppo.normalize_advantage,
        policy_kwargs=dict(net_arch=list(cfg.ppo.net_arch)),
        tensorboard_log=str(tensorboard_dir),
        seed=seed,
        device="cpu",
        verbose=1,
    )

    # ---- stage 2: behaviour cloning on model.policy ----
    print(f"\nBehaviour cloning ({args.bc_epochs} epochs, lr={args.bc_lr})...")
    behavior_clone(
        model, dataset,
        epochs=args.bc_epochs, batch_size=args.bc_batch_size, lr=args.bc_lr,
        vf_coef=args.bc_vf_coef, ent_coef=args.bc_ent_coef,
        max_grad_norm=cfg.ppo.max_grad_norm, val_fraction=args.bc_val_fraction, seed=seed,
    )
    bc_checkpoint = checkpoint_dir / "post_bc_model"
    model.save(str(bc_checkpoint))
    print(f"Saved post-BC (pre-finetune) checkpoint: {bc_checkpoint}.zip "
          "(evaluate this directly to see what BC alone achieves, before any PPO gradient)")

    # ---- stage 3: PPO fine-tuning ----
    checkpoint_callback = CheckpointCallback(
        save_freq=max(cfg.training.checkpoint_freq_steps // cfg.training.num_envs, 1),
        save_path=str(checkpoint_dir), name_prefix="ppo_edge_scheduler",
    )
    baseline_eval_callback = BaselineEvalCallback(
        eval_env=eval_env, greedy=GreedyScheduler(),
        eval_freq=max(cfg.training.eval_freq_steps // cfg.training.num_envs, 1),
        eval_suite=eval_suite, checkpoint_dir=checkpoint_dir, verbose=1,
    )

    print(f"\nPPO fine-tuning for {total_timesteps} timesteps "
          f"({cfg.training.num_envs} envs x {cfg.ppo.n_steps} steps/rollout = "
          f"{buffer_size} steps/update)...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([checkpoint_callback, baseline_eval_callback]),
        tb_log_name=f"maskable_ppo_{args.run_name}",
    )

    final_path = checkpoint_dir / "final_model"
    model.save(str(final_path))
    print("\nTraining complete.")
    print(f"Post-BC model:  {bc_checkpoint}.zip  (before any PPO gradient)")
    print(f"Final model:    {final_path}.zip")
    print(f"Best model:     {checkpoint_dir / 'best_model'}.zip  (by held-out eval reward)")
    rel = (checkpoint_dir / "best_model").relative_to(PROJECT_ROOT)
    print(f"Next:           python -m experiments.evaluate_agents --include-unseen "
          f"--model {rel} --seed-base 90000 --episodes-per-condition 20 "
          f"--out results/evaluation_{args.run_name}")


if __name__ == "__main__":
    main()