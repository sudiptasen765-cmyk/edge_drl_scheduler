"""
edge_environment.py

Ties together EdgeServer, Network, and TaskGenerator into a single
scheduler-agnostic simulation engine.

This class deliberately contains NO scheduling policy and NO reward
function. It only simulates: task arrivals, network transfer delay, server
queuing/execution, and bookkeeping (completions, failures, per-user stats).

Any scheduler - FIFO, Round Robin, Greedy, or a trained DQN agent - plugs
into this environment through the exact same interface:
    - next_pending_task()   : "what task needs a decision right now?"
    - submit_decision(...)  : "here's the server I chose for it"
    - advance_time(...)     : "move the simulation clock forward"

This separation is what makes the Stage 10/11 comparison between baseline
algorithms and the proposed DRL scheduler fair: every algorithm sees
identical server behavior, identical network behavior, and identical task
streams (same seed = same tasks).

Lifecycle of a task in this environment:
    1. Task arrives (per TaskGenerator's Poisson process) -> pending_queue
    2. Scheduler picks a server for it -> submit_decision()
    3. Task enters "network transfer" state (data is "in flight")
       -> pending_transfers
    4. Once transfer completes, task enters the target server's queue
       -> EdgeServer.assign_task()
    5. Server promotes it to running once CPU/RAM are free, executes it,
       and eventually marks it finished or (if deadline passed) failed.
"""

from environment.edge_server import EdgeServer
from environment.network import Network
from environment.task_generator import TaskGenerator, Task


class EdgeEnvironment:
    def __init__(
        self,
        server_configs: list[dict],
        num_users: int = 20,
        time_step_ms: float = 10.0,
        seed: int | None = None,
    ):
        """
        server_configs: list of dicts, each matching EdgeServer's constructor
                         keyword arguments (this is exactly the structure
                         loaded from config/server_config.yaml).
        time_step_ms:   simulation clock granularity for advance_time().
                         Smaller = more precise timing, more steps needed.
        """
        self.servers: dict[int, EdgeServer] = {
            cfg["server_id"]: EdgeServer(**cfg) for cfg in server_configs
        }
        self.network = Network()
        self.task_generator = TaskGenerator(num_users=num_users, seed=seed)
        self.num_users = num_users
        self.time_step_ms = time_step_ms

        # Episode state - populated by reset()
        self.tasks: list[Task] = []
        self.task_lookup: dict[int, Task] = {}
        self.duration_ms: float = 0.0
        self.current_time_ms: float = 0.0
        self.task_pointer: int = 0
        self.pending_queue: list[Task] = []
        self.pending_transfers: list[dict] = []
        self.completed_tasks: list[dict] = []
        self.failed_tasks: list[dict] = []
        self.user_stats: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Episode setup
    # ------------------------------------------------------------------

    def reset(
        self,
        workload_type: str = "normal",
        network_scenario: str = "normal",
        duration_ms: float = 10000.0,
        seed: int | None = None,
    ) -> list[Task]:
        """
        Start a new episode. Resets all servers, network, and generates a
        fresh task stream. Returns any tasks that arrive at t=0.
        """
        for server in self.servers.values():
            server.reset()
        self.network.reset()
        self.network.set_scenario(network_scenario)

        if seed is not None:
            self.task_generator.reset(seed=seed)
        else:
            self.task_generator.reset()

        self.tasks = self.task_generator.generate_episode(duration_ms, workload_type)
        self.task_lookup = {t.task_id: t for t in self.tasks}
        self.duration_ms = duration_ms
        self.current_time_ms = 0.0
        self.task_pointer = 0
        self.pending_queue = []
        self.pending_transfers = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.user_stats = {}

        return self._release_arrived_tasks()

    def _get_user_stats(self, user_id: int) -> dict:
        if user_id not in self.user_stats:
            self.user_stats[user_id] = {
                "tasks_submitted": 0,
                "tasks_completed": 0,
                "tasks_failed": 0,
                "last_served_time_ms": 0.0,
            }
        return self.user_stats[user_id]

    def _release_arrived_tasks(self) -> list[Task]:
        """Move any tasks whose arrival_time has passed into pending_queue."""
        newly_arrived = []
        while (
            self.task_pointer < len(self.tasks)
            and self.tasks[self.task_pointer].arrival_time_ms <= self.current_time_ms
        ):
            task = self.tasks[self.task_pointer]
            self.pending_queue.append(task)
            self._get_user_stats(task.user_id)["tasks_submitted"] += 1
            newly_arrived.append(task)
            self.task_pointer += 1
        return newly_arrived

    # ------------------------------------------------------------------
    # Scheduling interface (used by baselines and the DRL agent alike)
    # ------------------------------------------------------------------

    def has_pending_decision(self) -> bool:
        return len(self.pending_queue) > 0

    def next_pending_task(self) -> Task | None:
        """The task a scheduler needs to make a decision for right now."""
        return self.pending_queue[0] if self.pending_queue else None

    def get_in_flight_counts(self) -> dict[int, int]:
        """
        Number of tasks per server that have already been DECIDED (assigned
        by a scheduler via submit_decision) but have not yet landed in that
        server's actual queue - they're still in network transit
        (pending_transfers).

        This exists because a server's queue_length only updates once a
        transfer completes (in advance_time()), but multiple tasks can be
        decided in the same tick, before advance_time() ever runs. Without
        this, every decision in that tick sees an identical, stale picture
        of server load - as if none of the scheduler's own prior decisions
        in that tick happened - which causes load-aware schedulers (and,
        eventually, the DRL agent's observations) to pile many consecutive
        tasks onto whichever server currently looks emptiest.
        """
        counts = {sid: 0 for sid in self.servers}
        for entry in self.pending_transfers:
            counts[entry["server_id"]] += 1
        return counts

    def get_server_states(self) -> dict[int, dict]:
        """
        State of every server - the raw material for a scheduler's
        decision (rule-based or, later, the DRL agent's observation).

        Includes both the server's own reported state AND in-flight load
        (tasks already assigned to this server this "wave" but not yet
        arrived), merged into 'effective_queue_length'. Any scheduler doing
        load comparisons should use 'effective_queue_length', not the raw
        'queue_length', or it will be blind to its own just-made decisions.
        """
        in_flight = self.get_in_flight_counts()
        states = {}
        for sid, server in self.servers.items():
            state = server.get_state()
            state["in_flight_tasks"] = in_flight.get(sid, 0)
            state["effective_queue_length"] = state["queue_length"] + state["in_flight_tasks"]
            states[sid] = state
        return states

    def submit_decision(self, task: Task, server_id: int) -> bool:
        """
        Scheduler calls this with the task returned by next_pending_task()
        and the server_id it has chosen. Returns True if the task was
        accepted into the network-transfer pipeline, False if rejected
        (invalid server, or server has no room - counted as a failure).
        """
        if self.pending_queue and self.pending_queue[0].task_id == task.task_id:
            self.pending_queue.pop(0)
        else:
            raise ValueError(
                "submit_decision() must be called with the task returned by "
                "next_pending_task() - out-of-order submission is not allowed."
            )

        server = self.servers.get(server_id)
        if server is None:
            self._record_failure(task, reason="invalid_server_id")
            return False

        if not server.can_accept(task.cpu_requirement, task.ram_requirement):
            self._record_failure(task, reason="server_full_or_incompatible")
            return False

        # Task accepted - begins its network transfer phase.
        self.network.start_transfer(server_id)
        transfer_time_ms = self.network.total_network_delay_ms(task.data_size_mb, server)
        self.pending_transfers.append({
            "server_id": server_id,
            "task": task,
            "ready_time_ms": self.current_time_ms + transfer_time_ms,
        })
        return True

    def _record_failure(self, task: Task, reason: str):
        self._get_user_stats(task.user_id)["tasks_failed"] += 1
        self.failed_tasks.append({
            "task_id": task.task_id,
            "user_id": task.user_id,
            "reason": reason,
            "time_ms": self.current_time_ms,
        })

    # ------------------------------------------------------------------
    # Time-stepping
    # ------------------------------------------------------------------

    def advance_time(self, delta_ms: float | None = None) -> dict:
        """
        Move the simulation clock forward, resolving in-flight network
        transfers and stepping every server. Call this after all pending
        decisions for the current time have been submitted.
        """
        delta = delta_ms if delta_ms is not None else self.time_step_ms
        self.current_time_ms += delta

        # Resolve transfers that have fully arrived at their server.
        still_pending = []
        for entry in self.pending_transfers:
            if entry["ready_time_ms"] <= self.current_time_ms:
                self.network.end_transfer(entry["server_id"])
                server = self.servers[entry["server_id"]]
                task = entry["task"]
                accepted = server.assign_task(
                    task_id=task.task_id,
                    user_id=task.user_id,
                    cpu_requirement=task.cpu_requirement,
                    ram_requirement=task.ram_requirement,
                    execution_time_ms=task.execution_time_ms,
                    deadline_ms=task.deadline_ms,
                    current_time_ms=self.current_time_ms,
                )
                if not accepted:
                    # Queue filled up while this task was in transit.
                    self._record_failure(task, reason="queue_full_on_arrival")
            else:
                still_pending.append(entry)
        self.pending_transfers = still_pending

        # Step every server forward and collect completions/failures.
        step_summaries = []
        for server in self.servers.values():
            summary = server.step(time_delta_ms=delta, current_time_ms=self.current_time_ms)
            step_summaries.append(summary)

            for task_id in summary["finished_task_ids"]:
                task = self.task_lookup[task_id]
                stats = self._get_user_stats(task.user_id)
                stats["tasks_completed"] += 1
                stats["last_served_time_ms"] = self.current_time_ms
                self.completed_tasks.append({
                    "task_id": task_id,
                    "user_id": task.user_id,
                    "server_id": server.server_id,
                    "arrival_time_ms": task.arrival_time_ms,
                    "completion_time_ms": self.current_time_ms,
                    "latency_ms": self.current_time_ms - task.arrival_time_ms,
                    "deadline_ms": task.deadline_ms,
                    "met_deadline": self.current_time_ms <= task.deadline_ms,
                })

            for task_id in summary["failed_task_ids"]:
                task = self.task_lookup[task_id]
                self._get_user_stats(task.user_id)["tasks_failed"] += 1
                self.failed_tasks.append({
                    "task_id": task_id,
                    "user_id": task.user_id,
                    "server_id": server.server_id,
                    "reason": "deadline_missed_while_running",
                    "time_ms": self.current_time_ms,
                })

        newly_arrived = self._release_arrived_tasks()

        return {
            "current_time_ms": self.current_time_ms,
            "newly_arrived_tasks": newly_arrived,
            "server_summaries": step_summaries,
        }

    def is_episode_done(self) -> bool:
        """
        Episode ends when: all tasks have been generated and released, no
        decisions are pending, nothing is in network transit, and every
        server has fully drained its queue and running tasks.
        """
        return (
            self.current_time_ms >= self.duration_ms
            and self.task_pointer >= len(self.tasks)
            and not self.pending_queue
            and not self.pending_transfers
            and all(
                s.queue_length() == 0 and len(s.running_tasks) == 0
                for s in self.servers.values()
            )
        )

    # ------------------------------------------------------------------
    # Convenience: run a full episode with a given scheduler function
    # ------------------------------------------------------------------

    def run_episode(
        self,
        scheduler_fn,
        workload_type: str = "normal",
        network_scenario: str = "normal",
        duration_ms: float = 10000.0,
        seed: int | None = None,
        max_ticks: int = 100000,
    ) -> dict:
        """
        scheduler_fn: callable(task: Task, env: EdgeEnvironment) -> int
                      Given a task needing a decision and the environment
                      (for reading server states), returns a server_id.

        This is the entry point Stage 4 (testing without ML) and Stage 5
        (baseline schedulers) will use. The DRL training loop in Stage 7+
        will use the lower-level reset/next_pending_task/submit_decision/
        advance_time methods directly instead, since it needs to observe
        state and compute rewards between each individual decision.
        """
        self.reset(workload_type, network_scenario, duration_ms, seed)

        ticks = 0
        while not self.is_episode_done():
            while self.has_pending_decision():
                task = self.next_pending_task()
                server_id = scheduler_fn(task, self)
                self.submit_decision(task, server_id)
            self.advance_time()
            ticks += 1
            if ticks > max_ticks:
                raise RuntimeError(
                    "run_episode exceeded max_ticks - possible infinite loop "
                    "(e.g. a server that never drains its queue)."
                )

        return self.get_episode_summary()

    def get_episode_summary(self) -> dict:
        """
        Basic episode-level metrics. Full evaluation metrics (Jain's
        Fairness Index, throughput, etc.) are computed in metrics/ - this
        is just enough to sanity-check that an episode ran correctly.
        """
        total_tasks = len(self.tasks)
        num_completed = len(self.completed_tasks)
        num_failed = len(self.failed_tasks)
        avg_latency_ms = (
            sum(t["latency_ms"] for t in self.completed_tasks) / num_completed
            if num_completed > 0 else 0.0
        )
        deadline_met_count = sum(1 for t in self.completed_tasks if t["met_deadline"])
        total_energy_joules = sum(s.total_energy_joules for s in self.servers.values())

        return {
            "total_tasks_generated": total_tasks,
            "num_completed": num_completed,
            "num_failed": num_failed,
            "completion_rate": num_completed / total_tasks if total_tasks > 0 else 0.0,
            "avg_latency_ms": avg_latency_ms,
            "deadline_met_rate": deadline_met_count / num_completed if num_completed > 0 else 0.0,
            "total_energy_joules": total_energy_joules,
            "final_time_ms": self.current_time_ms,
        }