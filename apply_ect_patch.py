"""
apply_ect_patch.py - adds the ECT baseline to experiments/evaluate_agents.py.

    python apply_ect_patch.py            (run from D:\\edge_drl_scheduler)

Makes two edits and REFUSES to change anything if either expected line is not
found exactly once. A backup is written to experiments/evaluate_agents.py.bak.
Safe to run twice (it detects that ECT is already present).
"""
import shutil
import sys
from pathlib import Path

path = Path("experiments/evaluate_agents.py")
if not path.exists():
    sys.exit("Run this from the project root (experiments/evaluate_agents.py not found).")
text = path.read_text(encoding="utf-8")

if "EarliestCompletionScheduler" in text:
    sys.exit("Already patched - nothing to do.")

IMPORT_ANCHOR = "from scheduling.fifo import FIFOScheduler"
FACTORY_ANCHOR = '    "Greedy": lambda seed: GreedyScheduler(),'
for anchor in (IMPORT_ANCHOR, FACTORY_ANCHOR):
    if text.count(anchor) != 1:
        sys.exit(f"Expected exactly one occurrence of:\n  {anchor}\nfound {text.count(anchor)}. No changes made.")

text = text.replace(
    IMPORT_ANCHOR,
    "from scheduling.ect_scheduler import EarliestCompletionScheduler\n" + IMPORT_ANCHOR,
)
text = text.replace(
    FACTORY_ANCHOR,
    FACTORY_ANCHOR + '\n    "ECT": lambda seed: EarliestCompletionScheduler(),',
)
shutil.copy(path, path.with_suffix(".py.bak"))
path.write_text(text, encoding="utf-8")
print("Patched experiments/evaluate_agents.py (backup: evaluate_agents.py.bak).")
print("ECT is now one of the baselines; the paired table will include PPO minus ECT.")