"""
training/train_ppo.py

Phase 6: trains the first working DRL scheduler using masked PPO
(sb3-contrib's MaskablePPO) on top of the Gymnasium environment built in
Phase 5 (environment/edge_scheduling_env.py).

Usage:
    python -m training.train_ppo
    python -m training.train_ppo --timesteps 2000            (quick smoke test)
    python -m training.train_ppo --config config/ppo_config.yaml
    python -m training.train_ppo --no-reward-norm            (ablation)
    python -m training.train_ppo --eval-per-condition 3      (more eval episodes)
    python -m training.train_ppo --run-name focused --train-workloads normal variable
    python -m training.train_ppo --run-name ent01 --ent-coef 0.01
    python -m training.train_ppo --run-name focused_s2 --train-workloads normal variable --seed 2

What changed vs. the first version (and why)
--------------------------------------------
1. STRATIFIED EVALUATION.  The old callback evaluated on 5 fixed seeds and let
   `reset(seed=...)` DRAW the workload/network condition.  Those 5 seeds
   happened to land on 5/5 `congested` and 3/5 `heavy` - the hardest corner of
   the distribution - so Greedy itself scored ~6% completion and the
   "reward gap vs Greedy" curve was noise from an unlucky sample, not signal.
   The eval suite now pins the condition EXPLICITLY: every
   (workload x network) pair gets the same number of episodes, each with a
   fixed task_seed.  Metrics are equal-weighted across conditions.

2. GREEDY IS EVALUATED ONCE.  Greedy is deterministic and the suite is fixed,
   so its results never change between evaluations.  They are computed once
   when training starts and cached, which halves the evaluation cost.

3. PER-CONDITION LOGGING.  `eval/gap_<workload>_<network>` in TensorBoard
   shows WHERE the policy beats / loses to Greedy instead of one averaged
   number.

4. REWARD NORMALIZATION (VecNormalize, reward only).  The first run had
   value_loss in the thousands and explained_variance ~ 0 for 60k+ steps.
   With un-normalized returns of magnitude ~50-100 the value-function
   gradient dominates the *global* gradient-norm clip (max_grad_norm), which
   also throttles the policy gradient.  Rewards are now scaled by a running
   estimate of the return's std.  Observations are NOT normalized (the
   StateBuilder already bounds them to a fixed range), so saved models can be
   loaded and used for inference without any VecNormalize statistics.
   Disable with --no-reward-norm to ablate.

5. Monitor wrapper, so TensorBoard gets `rollout/ep_rew_mean` and
   `rollout/ep_len_mean` (RAW, un-normalized training-episode return).
"""

from __future__ import annotations

import argparse
import math
import random
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.ppo_config import PPOConfig
from environment.edge_scheduling_env import EdgeSchedulingEnv, rollout_with_scheduler
from scheduling.greedy_scheduler import GreedyScheduler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PPO_CONFIG = PROJECT_ROOT / "config" / "ppo_config.yaml"

# Fallbacks if the env config object does not expose its condition lists.
# They match the conditions exercised in test_drl_env.py.
DEFAULT_WORKLOADS = ("normal", "heavy", "burst", "variable")
DEFAULT_NETWORKS = ("normal", "congested")

# Distinct from any training seed range and from the seeds in evaluate_agents.py.
EVAL_TASK_SEED_BASE = 10_000


# ----------------------------------------------------------------------
# setup helpers
# ----------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch. MaskablePPO is also given `seed`
    directly (below), which seeds its own internal RNG for action sampling
    and rollout buffer shuffling; this covers everything else."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_overrides: dict | None = None) -> Monitor:
    """Factory for one training sub-environment (module-level function used via
    functools.partial, so it stays picklable if this is later switched to
    SubprocVecEnv). Monitor records the RAW episode return/length for TensorBoard.
    `env_overrides` is forwarded to EdgeSchedulingEnv.from_config_files()."""
    return Monitor(EdgeSchedulingEnv.from_config_files(env_overrides=env_overrides))


def build_vec_env(num_envs: int, env_overrides: dict | None = None) -> DummyVecEnv:
    return DummyVecEnv([partial(make_env, env_overrides) for _ in range(num_envs)])


def training_conditions(env: EdgeSchedulingEnv) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The (workloads, networks) the env samples from during training.
    Tries the env's config object first, falls back to the defaults above."""
    cfg = getattr(env, "env_config", None) or getattr(env, "config", None)
    workloads = tuple(getattr(cfg, "workloads", None) or DEFAULT_WORKLOADS)
    networks = tuple(getattr(cfg, "network_scenarios", None) or DEFAULT_NETWORKS)
    return workloads, networks


def build_eval_suite(
    workloads: tuple[str, ...],
    networks: tuple[str, ...],
    episodes_per_condition: int,
    seed_base: int = EVAL_TASK_SEED_BASE,
) -> list[dict]:
    """A fixed, balanced list of eval episodes.

    Every (workload, network) pair gets `episodes_per_condition` episodes with
    the condition pinned explicitly via reset options - never left to the
    env's random draw. Returns dicts: {"seed", "options", "condition"}.
    """
    suite = []
    for workload, network in product(workloads, networks):
        for i in range(episodes_per_condition):
            seed = seed_base + i
            suite.append({
                "seed": seed,
                "condition": f"{workload}_{network}",
                "options": {
                    "workload_type": workload,
                    "network_scenario": network,
                    "task_seed": seed,
                },
            })
    return suite


# ----------------------------------------------------------------------
# evaluation callback
# ----------------------------------------------------------------------

class BaselineEvalCallback(BaseCallback):
    """
    Every `eval_freq` callback calls (== eval_freq env-steps per parallel env,
    i.e. eval_freq * num_envs real environment steps), run the CURRENT policy
    on a fixed, condition-balanced eval suite and compare it, episode by
    episode, with GreedyScheduler on the identical episodes.

    Greedy's results are computed once (its behaviour on a fixed suite never
    changes) and cached.

    Logged to TensorBoard:
        eval/mean_reward, eval/mean_completion_rate
        eval/greedy_mean_reward, eval/greedy_mean_completion_rate
        eval/reward_gap_vs_greedy      (policy - greedy, mean over the suite)
        eval/win_rate_vs_greedy        (fraction of episodes policy > greedy)
        eval/gap_<workload>_<network>  (per-condition gap)

    Saves `checkpoint_dir/best_model` whenever the suite-mean policy reward
    improves.
    """

    def __init__(
        self,
        eval_env: EdgeSchedulingEnv,
        greedy: GreedyScheduler,
        eval_freq: int,
        eval_suite: list[dict],
        checkpoint_dir: Path,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.greedy = greedy
        self.eval_freq = eval_freq
        self.eval_suite = eval_suite
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_mean_reward = -float("inf")
        self._greedy_cache: list[dict] | None = None

    # -- Greedy: run once ------------------------------------------------
    def _compute_greedy_cache(self) -> list[dict]:
        results = []
        for ep in self.eval_suite:
            r = rollout_with_scheduler(
                self.eval_env, self.greedy, seed=ep["seed"], options=ep["options"]
            )
            results.append({
                "episode_return": float(r["episode_return"]),
                "completion_rate": float(r["completion_rate"]),
            })
        return results

    def _on_training_start(self) -> None:
        print(f"Evaluating Greedy once on the {len(self.eval_suite)}-episode eval suite...")
        self._greedy_cache = self._compute_greedy_cache()
        g = np.mean([r["episode_return"] for r in self._greedy_cache])
        gc = np.mean([r["completion_rate"] for r in self._greedy_cache])
        print(f"  Greedy: mean_return={g:.2f}  completion={gc * 100:.1f}%  (cached)")

    # -- policy episode --------------------------------------------------
    def _run_policy_episode(self, seed: int, options: dict) -> dict:
        """One deterministic episode of the current policy, using action
        masks exactly as MaskablePPO expects them at inference time."""
        obs, _ = self.eval_env.reset(seed=seed, options=options)
        done = False
        info: dict = {}
        while not done:
            action, _ = self.model.predict(
                obs, action_masks=self.eval_env.action_masks(), deterministic=True
            )
            obs, _, terminated, truncated, info = self.eval_env.step(int(action))
            done = terminated or truncated
        return info["episode_summary"]

    # -- hooks -----------------------------------------------------------
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        # Always evaluate the final policy, so best_model exists even for short
        # runs (e.g. --timesteps 2000) where the periodic eval never fires.
        self._evaluate()

    def _evaluate(self) -> None:
        if self._greedy_cache is None:  # safety net if _on_training_start was skipped
            self._greedy_cache = self._compute_greedy_cache()

        pol_returns, pol_completion = [], []
        gaps_by_condition: dict[str, list[float]] = {}

        for ep, g in zip(self.eval_suite, self._greedy_cache):
            s = self._run_policy_episode(ep["seed"], ep["options"])
            ret = float(s["episode_return"])
            pol_returns.append(ret)
            pol_completion.append(float(s["completion_rate"]))
            gaps_by_condition.setdefault(ep["condition"], []).append(ret - g["episode_return"])

        greedy_returns = [g["episode_return"] for g in self._greedy_cache]
        greedy_completion = [g["completion_rate"] for g in self._greedy_cache]
        all_gaps = np.array(pol_returns) - np.array(greedy_returns)

        mean_reward = float(np.mean(pol_returns))
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_completion_rate", float(np.mean(pol_completion)))
        self.logger.record("eval/greedy_mean_reward", float(np.mean(greedy_returns)))
        self.logger.record("eval/greedy_mean_completion_rate", float(np.mean(greedy_completion)))
        self.logger.record("eval/reward_gap_vs_greedy", float(np.mean(all_gaps)))
        self.logger.record("eval/win_rate_vs_greedy", float(np.mean(all_gaps > 0)))
        for cond, gaps in gaps_by_condition.items():
            self.logger.record(f"eval/gap_{cond}", float(np.mean(gaps)))
        self.logger.dump(self.num_timesteps)

        if self.verbose:
            print(
                f"[eval @ {self.num_timesteps} steps] "
                f"policy={mean_reward:.2f} ({np.mean(pol_completion) * 100:.1f}% completion)  "
                f"greedy={np.mean(greedy_returns):.2f} ({np.mean(greedy_completion) * 100:.1f}% completion)  "
                f"gap={np.mean(all_gaps):+.2f}  wins={np.mean(all_gaps > 0) * 100:.0f}%"
            )
            worst = sorted(gaps_by_condition.items(), key=lambda kv: np.mean(kv[1]))[:2]
            print("  weakest conditions vs Greedy: "
                  + ", ".join(f"{c} ({np.mean(v):+.1f})" for c, v in worst))

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(str(self.checkpoint_dir / "best_model"))
            if self.verbose:
                print(f"  new best model saved (mean_reward={mean_reward:.2f})")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train masked PPO on the edge task-scheduling environment."
    )
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_PPO_CONFIG),
        help="Path to config/ppo_config.yaml (or a variant for a different experiment).",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override training.total_timesteps - e.g. --timesteps 2000 for a quick smoke test.",
    )
    parser.add_argument(
        "--eval-per-condition", type=int, default=None,
        help="Eval episodes per (workload, network) pair. Default: derived from "
             "training.eval_episodes in the config, spread evenly over all conditions "
             "(at least 1 each).",
    )
    parser.add_argument(
        "--no-reward-norm", action="store_true",
        help="Disable VecNormalize reward scaling (for ablation).",
    )
    parser.add_argument(
        "--train-workloads", nargs="+", default=None,
        help="Restrict TRAINING to these workloads (e.g. normal variable). Evaluation "
             "always uses the full env config, so results stay comparable.",
    )
    parser.add_argument(
        "--train-networks", nargs="+", default=None,
        help="Restrict TRAINING to these network scenarios. Evaluation is unaffected.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Save checkpoints under <checkpoint_dir>/<run-name>/ and log to "
             "TensorBoard as maskable_ppo_<run-name>, so experiments don't overwrite each other.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override training.seed (Python/NumPy/PyTorch/env/PPO). Use for multi-seed replication.",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=None,
        help="Override ppo.ent_coef (entropy bonus) for this run.",
    )
    args = parser.parse_args()

    cfg = PPOConfig.from_yaml(args.config)
    total_timesteps = args.timesteps if args.timesteps is not None else cfg.training.total_timesteps

    buffer_size = cfg.ppo.n_steps * cfg.training.num_envs
    if buffer_size % cfg.ppo.batch_size != 0:
        raise ValueError(
            f"ppo.n_steps ({cfg.ppo.n_steps}) * training.num_envs "
            f"({cfg.training.num_envs}) = {buffer_size}, which is not a "
            f"multiple of ppo.batch_size ({cfg.ppo.batch_size}). Fix "
            "config/ppo_config.yaml - a non-multiple silently drops part of "
            "every rollout from training."
        )

    seed = args.seed if args.seed is not None else cfg.training.seed
    set_global_seed(seed)

    checkpoint_dir = PROJECT_ROOT / cfg.training.checkpoint_dir
    if args.run_name:
        checkpoint_dir = checkpoint_dir / args.run_name
    log_dir = PROJECT_ROOT / cfg.training.log_dir
    tensorboard_dir = PROJECT_ROOT / cfg.training.tensorboard_log
    for d in (checkpoint_dir, log_dir, tensorboard_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Building {cfg.training.num_envs} training environment(s)...")
    env_section = {}
    if args.train_workloads:
        env_section["workloads"] = list(args.train_workloads)
    if args.train_networks:
        env_section["network_scenarios"] = list(args.train_networks)
    train_overrides = {"environment": env_section} if env_section else None
    if train_overrides:
        print(f"Training restricted to: {env_section}  (evaluation still uses ALL conditions)")
    vec_env = build_vec_env(cfg.training.num_envs, train_overrides)
    vec_env.seed(seed)
    if not args.no_reward_norm:
        # Reward only: observations are already bounded by the StateBuilder.
        vec_env = VecNormalize(
            vec_env, norm_obs=False, norm_reward=True, gamma=cfg.ppo.gamma
        )
        print("Reward normalization: ON (VecNormalize, norm_obs=False)")
    else:
        print("Reward normalization: OFF")

    eval_env = EdgeSchedulingEnv.from_config_files()

    # ---- stratified eval suite ----
    workloads, networks = training_conditions(eval_env)
    n_conditions = len(workloads) * len(networks)
    per_condition = args.eval_per_condition or max(
        1, math.ceil(cfg.training.eval_episodes / n_conditions)
    )
    eval_suite = build_eval_suite(workloads, networks, per_condition)
    print(
        f"Eval suite: {len(workloads)} workloads {list(workloads)} x "
        f"{len(networks)} networks {list(networks)} x {per_condition} episode(s) "
        f"= {len(eval_suite)} episodes (condition-balanced, fixed seeds)"
    )

    ent_coef = args.ent_coef if args.ent_coef is not None else cfg.ppo.ent_coef
    print(f"Building MaskablePPO model... (ent_coef={ent_coef})")
    model = MaskablePPO(
        policy=cfg.ppo.policy,
        env=vec_env,
        learning_rate=cfg.ppo.learning_rate,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_range=cfg.ppo.clip_range,
        ent_coef=ent_coef,
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

    checkpoint_callback = CheckpointCallback(
        save_freq=max(cfg.training.checkpoint_freq_steps // cfg.training.num_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_edge_scheduler",
    )
    baseline_eval_callback = BaselineEvalCallback(
        eval_env=eval_env,
        greedy=GreedyScheduler(),
        eval_freq=max(cfg.training.eval_freq_steps // cfg.training.num_envs, 1),
        eval_suite=eval_suite,
        checkpoint_dir=checkpoint_dir,
        verbose=1,
    )

    print(
        f"Training for {total_timesteps} timesteps "
        f"({cfg.training.num_envs} envs x {cfg.ppo.n_steps} steps/rollout = "
        f"{buffer_size} steps/update)..."
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([checkpoint_callback, baseline_eval_callback]),
        tb_log_name="maskable_ppo" + (f"_{args.run_name}" if args.run_name else ""),
    )

    final_path = checkpoint_dir / "final_model"
    model.save(str(final_path))
    print("\nTraining complete.")
    print(f"Final model:  {final_path}.zip")
    print(f"Best model:   {checkpoint_dir / 'best_model'}.zip  (by held-out eval reward)")
    print(f"TensorBoard:  tensorboard --logdir {tensorboard_dir}")
    rel = (checkpoint_dir / "best_model").relative_to(PROJECT_ROOT)
    out = f"results/evaluation_{args.run_name}" if args.run_name else "results/evaluation"
    print(f"Next:         python -m experiments.evaluate_agents --include-unseen "
          f"--model {rel} --out {out}")


if __name__ == "__main__":
    main()