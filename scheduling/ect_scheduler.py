"""
ect_scheduler.py

Earliest-Completion-Time (ECT) baseline scheduler.

Idea (a classic "minimum completion time" heuristic from grid/cloud task
scheduling): for every server that could take the task, ESTIMATE when the task
would finish there, and pick the server with the earliest estimate.

    estimated_completion = max(network_delay, backlog) + execution_time

    network_delay  the EXACT transfer time this task would suffer on that
                   server right now, including the extra congestion its own
                   transfer adds (same formula the simulator uses).
    backlog        estimated time for the server to drain all work already
                   committed to it (running + queued + still in network
                   transit), assuming its cores are used in parallel:
                       sum(remaining_time * cpu) / cpu_cores
    max(...)       while the task's data is travelling, the server keeps
                   working through its backlog, so the two overlap instead of
                   adding up.

Why this baseline exists
------------------------
Greedy (least loaded) looks only at CPU use and queue length. It cannot see
that a lightly loaded server may sit behind a slow network link, and in this
simulator the deadline budget does NOT include transfer time - so network
delay is one of the biggest causes of missed deadlines. ECT sees it.

It is a strong non-learning reference: it shows how much better than Greedy a
scheduler can do using only information that is available at decision time. A
learned agent that cannot match it has not yet learned the basics.

Servers that cannot physically fit the task, or whose queue (queued plus
in-flight) is already full, are skipped, exactly like the action mask in the
Gymnasium environment. If nothing is eligible, it falls back to the server
with the shortest effective queue, like Greedy does.

Deliberately NOT modelled: energy and load balance. ECT optimises finishing
time only, so it is a fair "what if we only cared about speed" reference.
"""

from __future__ import annotations

from collections import defaultdict


class EarliestCompletionScheduler:
    """Callable scheduler: scheduler(task, env) -> server_id, where env is the EdgeEnvironment."""

    @staticmethod
    def _network_delay_ms(env, task, server) -> float:
        # Exactly what submit_decision() would compute: start_transfer() first,
        # then time the transfer. The network is restored afterwards.
        env.network.start_transfer(server.server_id)
        try:
            return env.network.total_network_delay_ms(task.data_size_mb, server)
        finally:
            env.network.end_transfer(server.server_id)

    def estimate_completion_ms(self, task, env, server_id, inflight_work=None) -> float:
        """Estimated time from now until `task` would finish on `server_id`."""
        server = env.servers[server_id]
        if inflight_work is None:
            inflight_work = self._inflight_work(env)
        work = (
            sum(t.remaining_time_ms * t.cpu_requirement for t in server.running_tasks)
            + sum(t.remaining_time_ms * t.cpu_requirement for t in server.queue)
            + inflight_work.get(server_id, 0.0)
        )
        backlog_ms = work / server.cpu_cores if server.cpu_cores > 0 else float("inf")
        delay_ms = self._network_delay_ms(env, task, server)
        return max(delay_ms, backlog_ms) + task.execution_time_ms

    @staticmethod
    def _inflight_work(env) -> dict:
        work = defaultdict(float)
        for entry in env.pending_transfers:
            t = entry["task"]
            work[entry["server_id"]] += t.execution_time_ms * t.cpu_requirement
        return work

    def __call__(self, task, env) -> int:
        states = env.get_server_states()
        inflight_work = self._inflight_work(env)

        best_key, best_sid = None, None
        for sid in sorted(env.servers):
            server = env.servers[sid]
            fits = (
                task.cpu_requirement <= server.cpu_cores
                and task.ram_requirement <= server.ram_gb
            )
            if not fits or states[sid]["effective_queue_length"] >= server.max_queue_length:
                continue
            key = (self.estimate_completion_ms(task, env, sid, inflight_work), sid)
            if best_key is None or key < best_key:
                best_key, best_sid = key, sid

        if best_sid is not None:
            return best_sid

        # Nothing eligible: pick the least-bad option so the simulator can
        # record a deterministic failure (same fallback as GreedyScheduler).
        return min(states, key=lambda sid: (states[sid]["effective_queue_length"], sid))