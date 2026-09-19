"""
edge_server.py

Models a single heterogeneous edge server.

Each server tracks:
- Static capacity (CPU cores, RAM)
- Dynamic state (current utilization, active tasks, queue)
- Network characteristics (base latency, bandwidth)
- Energy consumption, using an idle + dynamic power model:

      P(u) = power_idle_w + (power_max_w - power_idle_w) * u^power_exponent

  where u is CPU utilization in [0, 1]. This captures two real effects that a
  purely linear model misses:
    1. Idle cost: a server draws non-zero power even with 0% CPU load
       (fans, RAM refresh, motherboard baseline).
    2. Superlinear scaling: as utilization approaches 100%, power draw grows
       faster than proportionally (voltage/frequency scaling, thermal
       effects). power_exponent > 1 encodes this; = 1 reduces to a linear
       model above the idle baseline.

A server does NOT decide which tasks it receives - that is the scheduler's
job (baseline algorithms or the DRL agent). The server only:
  - reports whether it CAN accept a given task (capacity check)
  - accepts a task if instructed to
  - advances its own simulation clock (processes queued/running tasks)
  - reports its own state (for the DRL agent's observation and for metrics)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RunningTask:
    """A task currently occupying resources on this server."""
    task_id: int
    user_id: int
    cpu_requirement: float      # fraction of a core, e.g. 0.5 = half a core
    ram_requirement: float      # GB
    remaining_time_ms: float    # time left until this task finishes executing
    deadline_ms: float          # absolute simulation time by which it must finish
    arrival_time_ms: float      # when the task arrived at this server's queue


class EdgeServer:
    """
    A single heterogeneous edge server.

    All capacity values (cpu_cores, ram_gb) represent the TOTAL resource pool.
    Utilization is derived, never stored directly, so it can never drift out
    of sync with what's actually running.
    """

    def __init__(
        self,
        server_id: int,
        cpu_cores: float,
        ram_gb: float,
        base_latency_ms: float,
        bandwidth_mbps: float,
        power_idle_w: float,
        power_max_w: float,
        power_exponent: float = 1.2,
        max_queue_length: int = 20,
    ):
        # --- Static configuration (does not change during simulation) ---
        self.server_id = server_id
        self.cpu_cores = cpu_cores
        self.ram_gb = ram_gb
        self.base_latency_ms = base_latency_ms
        self.bandwidth_mbps = bandwidth_mbps
        self.power_idle_w = power_idle_w
        self.power_max_w = power_max_w
        self.power_exponent = power_exponent
        self.max_queue_length = max_queue_length

        # --- Dynamic state ---
        self.running_tasks: list[RunningTask] = []   # tasks actively executing
        self.queue: list[RunningTask] = []            # tasks waiting for capacity

        # --- Bookkeeping / metrics (accumulated over the simulation) ---
        self.total_energy_joules: float = 0.0
        self.completed_task_count: int = 0
        self.failed_task_count: int = 0  # deadline missed or rejected
        self.total_tasks_served_per_user: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Utilization / capacity
    # ------------------------------------------------------------------

    def cpu_used(self) -> float:
        """Sum of CPU cores currently occupied by running tasks."""
        return sum(t.cpu_requirement for t in self.running_tasks)

    def ram_used(self) -> float:
        """Sum of RAM (GB) currently occupied by running tasks."""
        return sum(t.ram_requirement for t in self.running_tasks)

    def cpu_utilization(self) -> float:
        """CPU utilization as a fraction in [0, 1]."""
        if self.cpu_cores <= 0:
            return 0.0
        return min(1.0, self.cpu_used() / self.cpu_cores)

    def ram_utilization(self) -> float:
        """RAM utilization as a fraction in [0, 1]."""
        if self.ram_gb <= 0:
            return 0.0
        return min(1.0, self.ram_used() / self.ram_gb)

    def available_cpu(self) -> float:
        return max(0.0, self.cpu_cores - self.cpu_used())

    def available_ram(self) -> float:
        return max(0.0, self.ram_gb - self.ram_used())

    def queue_length(self) -> int:
        return len(self.queue)

    def can_accept(self, cpu_requirement: float, ram_requirement: float) -> bool:
        """
        Whether this server's QUEUE has room for a new task.

        Note: this checks queue capacity, not immediate CPU/RAM availability.
        A task can be accepted into the queue even if resources are busy right
        now - it will simply wait. This mirrors how real edge servers behave
        (they queue work rather than rejecting it purely because the CPU
        happens to be busy this instant). Immediate execution vs. queuing is
        decided in step().
        """
        if self.queue_length() >= self.max_queue_length:
            return False
        if cpu_requirement > self.cpu_cores:
            return False  # task physically cannot fit on this server, ever
        if ram_requirement > self.ram_gb:
            return False
        return True

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def assign_task(
        self,
        task_id: int,
        user_id: int,
        cpu_requirement: float,
        ram_requirement: float,
        execution_time_ms: float,
        deadline_ms: float,
        current_time_ms: float,
    ) -> bool:
        """
        Attempt to assign a task to this server's queue.
        Returns True if accepted, False if rejected (queue full or task
        physically cannot fit).
        """
        if not self.can_accept(cpu_requirement, ram_requirement):
            self.failed_task_count += 1
            return False

        task = RunningTask(
            task_id=task_id,
            user_id=user_id,
            cpu_requirement=cpu_requirement,
            ram_requirement=ram_requirement,
            remaining_time_ms=execution_time_ms,
            deadline_ms=deadline_ms,
            arrival_time_ms=current_time_ms,
        )
        self.queue.append(task)
        return True

    def step(self, time_delta_ms: float, current_time_ms: float) -> dict:
        """
        Advance simulation time on this server by time_delta_ms.

        Responsibilities:
          1. Promote queued tasks into running_tasks if capacity now allows.
          2. Decrease remaining_time_ms for all running tasks.
          3. Remove and count tasks that finished this step.
          4. Remove and count tasks that missed their deadline (failed).
          5. Accumulate energy consumption for this time slice.

        Returns a small summary dict, useful for logging/metrics without the
        caller needing to inspect internal task lists directly.
        """
        # --- 1. Try to promote queued tasks into execution ---
        still_queued = []
        for task in self.queue:
            if (
                task.cpu_requirement <= self.available_cpu()
                and task.ram_requirement <= self.available_ram()
            ):
                self.running_tasks.append(task)
            else:
                still_queued.append(task)
        self.queue = still_queued

        # --- 2 & 4. Advance running tasks, check deadlines ---
        finished_task_ids = []
        failed_task_ids = []
        still_running = []
        for task in self.running_tasks:
            task.remaining_time_ms -= time_delta_ms

            if task.remaining_time_ms <= 0:
                finished_task_ids.append(task.task_id)
                self.completed_task_count += 1
                self.total_tasks_served_per_user[task.user_id] = (
                    self.total_tasks_served_per_user.get(task.user_id, 0) + 1
                )
            elif current_time_ms > task.deadline_ms:
                # Still running but deadline has passed - count as failed.
                failed_task_ids.append(task.task_id)
                self.failed_task_count += 1
            else:
                still_running.append(task)
        self.running_tasks = still_running

        # --- 3. Energy accumulation for this time slice ---
        # Using utilization AFTER promotion/completion changes above, since
        # that reflects what was actually running during this slice.
        power_w = self.current_power_draw()
        # Energy (J) = Power (W) * time (s)
        self.total_energy_joules += power_w * (time_delta_ms / 1000.0)

        return {
            "server_id": self.server_id,
            "finished_task_ids": finished_task_ids,
            "failed_task_ids": failed_task_ids,
            "power_w": power_w,
            "cpu_utilization": self.cpu_utilization(),
            "ram_utilization": self.ram_utilization(),
            "queue_length": self.queue_length(),
        }

    # ------------------------------------------------------------------
    # Energy model
    # ------------------------------------------------------------------

    def current_power_draw(self) -> float:
        """
        Instantaneous power draw in Watts, using the idle + dynamic model:
            P(u) = power_idle_w + (power_max_w - power_idle_w) * u^power_exponent
        """
        u = self.cpu_utilization()
        dynamic_range = self.power_max_w - self.power_idle_w
        return self.power_idle_w + dynamic_range * (u ** self.power_exponent)

    # ------------------------------------------------------------------
    # Reset / state export
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all dynamic state. Static config (capacity, power curve) is kept."""
        self.running_tasks = []
        self.queue = []
        self.total_energy_joules = 0.0
        self.completed_task_count = 0
        self.failed_task_count = 0
        self.total_tasks_served_per_user = {}

    def get_state(self) -> dict:
        """
        Returns this server's observable state - the fields the DRL agent's
        state vector and the dashboard will both read from. Keeping this as
        one method means both consumers always see identical, consistent data.
        """
        return {
            "server_id": self.server_id,
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.ram_gb,
            "cpu_utilization": self.cpu_utilization(),
            "ram_utilization": self.ram_utilization(),
            "available_cpu": self.available_cpu(),
            "available_ram": self.available_ram(),
            "queue_length": self.queue_length(),
            "base_latency_ms": self.base_latency_ms,
            "bandwidth_mbps": self.bandwidth_mbps,
            "power_w": self.current_power_draw(),
            "total_energy_joules": self.total_energy_joules,
            "completed_task_count": self.completed_task_count,
            "failed_task_count": self.failed_task_count,
        }

    def __repr__(self):
        return (
            f"EdgeServer(id={self.server_id}, "
            f"cpu={self.cpu_utilization():.0%}, "
            f"ram={self.ram_utilization():.0%}, "
            f"queue={self.queue_length()}, "
            f"power={self.current_power_draw():.1f}W)"
        )