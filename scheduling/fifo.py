"""
fifo.py

FIFO / First-Fit baseline scheduler.

Tasks are already processed in arrival order by the environment (pending_queue
is a strict FIFO). This scheduler's job is simply: given the current task,
assign it to the FIRST server (lowest server_id) that currently has room.

This is the simplest possible "does the job" policy - no load awareness at
all beyond a basic capacity check. It exists as a baseline to show what
happens when scheduling ignores server load entirely.
"""


class FIFOScheduler:
    """Callable scheduler: fifo_scheduler(task, env) -> server_id"""

    def __call__(self, task, env) -> int:
        server_ids = sorted(env.servers.keys())
        for server_id in server_ids:
            server = env.servers[server_id]
            if server.can_accept(task.cpu_requirement, task.ram_requirement):
                return server_id
        # No server currently has room - return the first server anyway.
        # The environment will record this as a failure (server_full), which
        # is the correct outcome: FIFO has no fallback strategy for overload.
        return server_ids[0]