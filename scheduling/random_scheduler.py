"""
random_scheduler.py

Random baseline scheduler.

Picks a server uniformly at random for every task, with no regard for load,
capacity, or anything else. This is the "does no thinking at all" lower
bound - any scheduler (rule-based or learned) that fails to clearly beat
Random on the core metrics (latency, completion rate, fairness) is not
actually adding value.

Uses its own seeded RNG so results are reproducible independent of whatever
else is drawing random numbers elsewhere in the simulation.
"""

import numpy as np


class RandomScheduler:
    """Callable scheduler: random_scheduler(task, env) -> server_id"""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def __call__(self, task, env) -> int:
        server_ids = sorted(env.servers.keys())
        return int(self.rng.choice(server_ids))