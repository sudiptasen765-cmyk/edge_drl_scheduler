"""
patch_train_ppo_bc.py - two fixes to training/train_ppo_bc.py.   Run from the project root:

    python patch_train_ppo_bc.py

FIX 1 (per-seed demonstrations). BC_DATA_SEED_BASE was never combined with --seed, so every
      seed cloned ECT on the SAME episodes. Now: seed_base = BC_DATA_SEED_BASE + 10_000 * seed
      (disjoint for seeds 0..~19 and bc-episodes < 10,000; eval seeds are 10_000, 50_000, 90_000).

FIX 2 (value targets). During collection VecNormalize divides rewards by a RUNNING std that
      drifts by orders of magnitude (measured: 0.01 -> 52.6 within 300 steps), so the recorded
      returns were on wildly different scales in early vs late episodes. Now the dataset is built
      from the ORIGINAL rewards (VecNormalize.get_original_reward()) and the returns are divided
      once by the FINAL running std - the scale PPO fine-tuning starts from, because ret_rms is
      transferred to the fine-tuning env. (The per-step reward clip at +/-10 is not reproduced.)

All-or-nothing: if any expected line is not found exactly once, nothing is changed.
A backup is written to training/train_ppo_bc.py.bak.  Running it twice is detected.
"""
import shutil
import sys
from pathlib import Path

path = Path("training/train_ppo_bc.py")
if not path.exists():
    sys.exit("Run from the project root (training/train_ppo_bc.py not found).")
text = path.read_text(encoding="utf-8")
if "get_original_reward" in text:
    sys.exit("Already patched - nothing to do.")

edits = [
    # (anchor, replacement)
    ("            reward_list.append(float(reward[0]))",
     "            # ORIGINAL (un-normalised) reward: VecNormalize's divisor drifts during collection\n"
     "            reward_list.append(float(vec_env.get_original_reward()[0]) if use_reward_norm else float(reward[0]))"),
    ('    dataset = {\n        "obs": np.asarray(obs_list, dtype=np.float32),',
     "    if use_reward_norm:\n"
     "        # one common scale: the final running std, which is what fine-tuning starts from\n"
     "        returns = (returns / np.sqrt(vec_env.ret_rms.var + vec_env.epsilon)).astype(np.float32)\n\n"
     '    dataset = {\n        "obs": np.asarray(obs_list, dtype=np.float32),'),
    ("cfg, train_overrides, args.bc_episodes, use_reward_norm",
     "cfg, train_overrides, args.bc_episodes, use_reward_norm,\n"
     "        seed_base=BC_DATA_SEED_BASE + 10_000 * seed"),
    ("(seeds {BC_DATA_SEED_BASE}+)", "(seeds {BC_DATA_SEED_BASE + 10_000 * seed}+)"),
]
for anchor, _ in edits:
    n = text.count(anchor)
    if n != 1:
        sys.exit(f"Expected exactly one occurrence of:\n  {anchor!r}\nfound {n}. No changes made.")
for anchor, repl in edits:
    text = text.replace(anchor, repl)
import ast
ast.parse(text)                      # refuse to write a file that does not even parse
shutil.copy(path, path.with_suffix(".py.bak"))
path.write_text(text, encoding="utf-8")
print("Patched training/train_ppo_bc.py (backup: train_ppo_bc.py.bak).")
print("Demonstrations now differ per --seed; BC value targets use one consistent scale.")