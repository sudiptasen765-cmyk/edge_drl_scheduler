"""
greedy_scheduler.py

Greedy / Least-Loaded baseline scheduler.

Picks whichever server currently has the LOWEST combined load score,
computed from CPU utilization and EFFECTIVE queue length (queue length
normalized by that server's own max_queue_length, since queue capacity
differs across heterogeneous servers - a queue of 5 means something
different on a 15-slot server versus a 40-slot server).

    load_score = 0.5 * cpu_utilization + 0.5 * (effective_queue_length / max_queue_length)

Uses effective_queue_length (queue_length + in_flight_tasks), not raw
queue_length. Raw queue_length only updates once a task's network transfer
completes, so if it were used directly, every task decided within the same
tick (before advance_time() runs) would see an identical, stale picture -
as if none of Greedy's own prior decisions that tick had happened - and
pile onto whichever server looked emptiest at the start of the tick.

This is the strongest of the non-learning baselines: it's reactive (uses
only CURRENT state, no prediction) but load-aware, unlike FIFO/Round Robin/
Random. The DRL agent needs to meaningfully beat this to justify the added
complexity of prediction + fairness + explainability - if it can't, that's
an important (and honest) experimental finding, not something to hide.

Servers that cannot physically fit the task (or have a full queue) are
excluded from consideration entirely, rather than just scored poorly - a
task should never be routed to a server that will simply reject it if any
viable alternative exists.
"""


class GreedyScheduler:
    """Callable scheduler: greedy_scheduler(task, env) -> server_id"""

    def __call__(self, task, env) -> int:
        server_states = env.get_server_states()

        candidates = []
        for server_id, state in server_states.items():
            server = env.servers[server_id]
            if not server.can_accept(task.cpu_requirement, task.ram_requirement):
                continue
            queue_fraction = (
                state["effective_queue_length"] / server.max_queue_length
                if server.max_queue_length > 0 else 1.0
            )
            load_score = 0.5 * state["cpu_utilization"] + 0.5 * queue_fraction
            candidates.append((load_score, server_id))

        if not candidates:
            # No server can currently accept this task - fall back to the
            # least-bad option (lowest effective queue among all servers) so
            # the environment can still record a meaningful, deterministic
            # failure rather than an arbitrary one.
            server_id = min(
                server_states.keys(),
                key=lambda sid: server_states[sid]["effective_queue_length"],
            )
            return server_id

        candidates.sort(key=lambda c: c[0])
        return candidates[0][1]