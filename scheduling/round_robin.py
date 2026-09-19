"""
round_robin.py

Round Robin baseline scheduler.

Cycles through servers in a fixed rotation, completely ignoring current
load. Each new task goes to "whichever server is next in line," regardless
of whether that server is already overloaded or sitting idle.

This is a classic naive load-distribution baseline: it spreads tasks evenly
by COUNT, not by actual capacity - which is exactly its weakness when
servers are heterogeneous (a 2-core and a 12-core server get the same
number of tasks under Round Robin, even though they can handle very
different amounts of work).
"""


class RoundRobinScheduler:
    """Callable scheduler: round_robin_scheduler(task, env) -> server_id"""

    def __init__(self):
        self._index = 0

    def reset(self):
        """Call this at the start of each episode so comparisons across
        episodes/schedulers start from the same rotation position."""
        self._index = 0

    def __call__(self, task, env) -> int:
        server_ids = sorted(env.servers.keys())
        server_id = server_ids[self._index % len(server_ids)]
        self._index += 1
        return server_id