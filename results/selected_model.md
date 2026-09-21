# Selected PPO scheduler (Phase 6)

**File:** `models/ppo_edge_scheduler_selected.zip` (a copy of `models/ppo_checkpoints/focused_s3/best_model.zip`)

## How it was trained
- Algorithm: MaskablePPO (sb3-contrib 2.8.0, stable-baselines3 2.8.0, torch 2.6.0+cpu, gymnasium 0.29.1)
- Command: `python -m training.train_ppo --run-name focused_s3 --train-workloads normal variable --seed 3`
- Config: `config/ppo_config.yaml` (300,000 steps, 4 envs, net_arch [128, 128], reward-only VecNormalize)
- Training conditions: workloads `normal`, `variable`; both network scenarios. Evaluation always uses all conditions.

## How it was chosen
- 8 seeds (1-8) of the same setup were trained.
- Selection: PPO minus Greedy, mean over all 15 conditions, on task seeds 50000+ (10 episodes per condition), `experiments/select_model.py`. Seed 3 ranked first (+20.5).
- Reporting: task seeds 90000+ (20 episodes per condition), never used for any choice.
  Seed 3: **+18.1** vs Greedy (all-condition mean). Ranking of all 8 seeds was identical on both sets.

## Setup-level result (the headline)
All 8 seeds, PPO minus Greedy, held-out seeds 90000+, mean over all 15 conditions:
mean **+9.7**, sd 7.5, 95% CI [+3.4, +15.9], 6 of 8 seeds above Greedy.
By regime (Greedy completion < 15% = saturated): saturated +12.4 (8/8 seeds positive), headroom +5.6 (95% CI [-0.5, +11.7], 6/8 positive).

## Known limitations
- Two of eight seeds do not beat Greedy: training is not stable across seeds.
- Balance and energy cost about -10 vs Greedy in every run; the fleet takes about 2.4x longer to drain. The gain comes entirely from the SLA term.
- Training on heavy/burst workloads (`alltrain`, `allload`, `alltrainnonorm`) was worse than training on normal+variable only, on all conditions. Cause not identified; reward normalization was tested and is not the explanation (3 seeds each).
- The default evaluation seeds (50000+) and the selection seeds are the same set.