"""
task_generator.py

Generates task arrival streams for an episode of simulation.

Arrival timing:
    Task arrivals follow a Poisson process - the standard model for
    independent, randomly-timed arrivals in queueing/network simulation.
    For workloads with a CONSTANT rate (normal, heavy, unseen), this is a
    straightforward homogeneous Poisson process: inter-arrival times are
    drawn from an exponential distribution.

    For workloads with a rate that CHANGES over time (burst, variable), we
    use a non-homogeneous Poisson process, generated via the standard
    "thinning" (rejection sampling) algorithm:
        1. Find rate_max, the highest rate the process ever reaches.
        2. Generate candidate arrivals as if the rate were rate_max
           everywhere (a simple homogeneous process).
        3. Keep each candidate arrival with probability
           actual_rate(t) / rate_max, discard otherwise.
    This produces a correctly-distributed time-varying Poisson process
    without needing a closed-form inverse for the rate function.

Deadline generation:
    deadline_ms = arrival_time_ms + execution_time_ms * slack_factor

    slack_factor is drawn from a range that depends on task priority:
    high-priority tasks get a TIGHTER slack range (less room for error),
    low-priority tasks get a more generous one. This means priority isn't
    just a label - it directly determines how much scheduling pressure a
    task creates, which is what the fairness and deadline-violation metrics
    need in order to be meaningful.

Workload types:
    normal   - constant, moderate arrival rate
    heavy    - constant, high arrival rate
    burst    - normally low rate, with a few short high-rate spike windows
    variable - rate oscillates smoothly over the episode (sinusoidal)
    unseen   - constant rate, but task characteristics (size, resource
               requirements, priority mix) are deliberately drawn from a
               DIFFERENT distribution than the other four. This is used
               exclusively for generalization testing - the DRL agent must
               never train on this workload type.
"""

import math
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Task:
    """A single computational task submitted by a user."""
    task_id: int
    user_id: int
    cpu_requirement: float      # fraction of a core (or whole cores)
    ram_requirement: float      # GB
    data_size_mb: float         # input data size to transfer
    execution_time_ms: float    # expected execution time once running
    deadline_ms: float          # ABSOLUTE simulation time by which it must finish
    priority: str                # "high", "medium", "low"
    arrival_time_ms: float

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "cpu_requirement": self.cpu_requirement,
            "ram_requirement": self.ram_requirement,
            "data_size_mb": self.data_size_mb,
            "execution_time_ms": self.execution_time_ms,
            "deadline_ms": self.deadline_ms,
            "priority": self.priority,
            "arrival_time_ms": self.arrival_time_ms,
        }


# ----------------------------------------------------------------------
# Workload presets. Each defines the statistical characteristics of tasks
# and their arrival process. Kept as one place so every experiment that
# references "heavy" or "unseen" means exactly the same thing.
# ----------------------------------------------------------------------
WORKLOAD_PRESETS = {
    "normal": {
        "arrival_rate_per_sec": 5.0,
        "cpu_range": (0.5, 2.0),
        "ram_range": (0.5, 2.0),
        "data_size_range_mb": (1.0, 10.0),
        "exec_time_range_ms": (100.0, 800.0),
        "priority_weights": {"high": 0.2, "medium": 0.5, "low": 0.3},
    },
    "heavy": {
        "arrival_rate_per_sec": 18.0,
        "cpu_range": (0.5, 2.5),
        "ram_range": (0.5, 2.5),
        "data_size_range_mb": (1.0, 12.0),
        "exec_time_range_ms": (100.0, 900.0),
        "priority_weights": {"high": 0.25, "medium": 0.5, "low": 0.25},
    },
    "burst": {
        "base_rate_per_sec": 4.0,
        "burst_rate_per_sec": 35.0,
        "num_bursts": 3,
        "burst_window_ms": 1500.0,
        "cpu_range": (0.5, 2.0),
        "ram_range": (0.5, 2.0),
        "data_size_range_mb": (1.0, 10.0),
        "exec_time_range_ms": (100.0, 800.0),
        "priority_weights": {"high": 0.2, "medium": 0.5, "low": 0.3},
    },
    "variable": {
        "base_rate_per_sec": 8.0,
        "amplitude_per_sec": 6.0,
        "period_ms": 6000.0,
        "cpu_range": (0.5, 2.0),
        "ram_range": (0.5, 2.0),
        "data_size_range_mb": (1.0, 10.0),
        "exec_time_range_ms": (100.0, 800.0),
        "priority_weights": {"high": 0.2, "medium": 0.5, "low": 0.3},
    },
    "unseen": {
        # Deliberately different from all training workloads: larger and
        # more variable tasks, higher-resource requirements, a heavier tilt
        # toward high-priority tasks. The DRL agent should never see this
        # during training - it exists only to test generalization.
        "arrival_rate_per_sec": 12.0,
        "cpu_range": (1.0, 6.0),
        "ram_range": (1.0, 8.0),
        "data_size_range_mb": (5.0, 50.0),
        "exec_time_range_ms": (200.0, 2000.0),
        "priority_weights": {"high": 0.4, "medium": 0.4, "low": 0.2},
    },
}

# Deadline slack ranges by priority - tighter for higher priority.
SLACK_RANGES = {
    "high": (1.2, 2.0),
    "medium": (1.8, 3.0),
    "low": (2.5, 4.0),
}


class TaskGenerator:
    """
    Generates a full episode's worth of tasks for a given workload type.
    """

    def __init__(self, num_users: int = 20, seed: int | None = None):
        self.num_users = num_users
        self.rng = np.random.default_rng(seed)
        self._next_task_id = 1

    def reset(self, seed: int | None = None):
        """Reset task ID counter and optionally reseed the RNG."""
        self._next_task_id = 1
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_episode(self, duration_ms: float, workload_type: str) -> list[Task]:
        """
        Generate all tasks that arrive during [0, duration_ms) under the
        given workload type. Returns tasks sorted by arrival_time_ms.
        """
        if workload_type not in WORKLOAD_PRESETS:
            raise ValueError(
                f"Unknown workload type '{workload_type}'. "
                f"Available: {list(WORKLOAD_PRESETS.keys())}"
            )
        preset = WORKLOAD_PRESETS[workload_type]

        arrival_times = self._generate_arrival_times(duration_ms, workload_type, preset)

        tasks = [
            self._generate_task(arrival_time_ms=t, preset=preset)
            for t in arrival_times
        ]
        tasks.sort(key=lambda t: t.arrival_time_ms)
        return tasks

    # ------------------------------------------------------------------
    # Arrival time generation
    # ------------------------------------------------------------------

    def _generate_arrival_times(
        self, duration_ms: float, workload_type: str, preset: dict
    ) -> list[float]:
        if workload_type in ("normal", "heavy", "unseen"):
            return self._homogeneous_poisson_arrivals(
                duration_ms, rate_per_ms=preset["arrival_rate_per_sec"] / 1000.0
            )

        if workload_type == "burst":
            return self._burst_arrivals(duration_ms, preset)

        if workload_type == "variable":
            return self._variable_arrivals(duration_ms, preset)

        raise ValueError(f"No arrival generator implemented for '{workload_type}'")

    def _homogeneous_poisson_arrivals(self, duration_ms: float, rate_per_ms: float) -> list[float]:
        """Standard Poisson process: exponential inter-arrival times."""
        if rate_per_ms <= 0:
            return []
        arrivals = []
        t = 0.0
        while True:
            inter_arrival = self.rng.exponential(1.0 / rate_per_ms)
            t += inter_arrival
            if t >= duration_ms:
                break
            arrivals.append(t)
        return arrivals

    def _burst_arrivals(self, duration_ms: float, preset: dict) -> list[float]:
        """
        Base low rate everywhere, with a few short high-rate burst windows
        placed at random times. Implemented via thinning: generate
        candidates at the maximum rate (burst_rate), keep each with
        probability actual_rate(t) / burst_rate.
        """
        base_rate = preset["base_rate_per_sec"] / 1000.0
        burst_rate = preset["burst_rate_per_sec"] / 1000.0
        window_ms = preset["burst_window_ms"]
        num_bursts = preset["num_bursts"]

        # Randomly place burst windows within the episode duration.
        burst_starts = sorted(
            self.rng.uniform(0, max(1.0, duration_ms - window_ms), size=num_bursts)
        )
        burst_windows = [(s, s + window_ms) for s in burst_starts]

        def rate_at(t):
            for start, end in burst_windows:
                if start <= t <= end:
                    return burst_rate
            return base_rate

        return self._thinned_arrivals(duration_ms, rate_max=burst_rate, rate_fn=rate_at)

    def _variable_arrivals(self, duration_ms: float, preset: dict) -> list[float]:
        """
        Sinusoidally varying rate: rate(t) = base + amplitude * sin(2*pi*t/period).
        Floored at a small positive value so the rate never goes negative
        (and arrivals never fully stop).
        """
        base = preset["base_rate_per_sec"] / 1000.0
        amplitude = preset["amplitude_per_sec"] / 1000.0
        period = preset["period_ms"]
        rate_max = base + amplitude  # peak of the sinusoid

        def rate_at(t):
            return max(0.05 / 1000.0, base + amplitude * math.sin(2 * math.pi * t / period))

        return self._thinned_arrivals(duration_ms, rate_max=rate_max, rate_fn=rate_at)

    def _thinned_arrivals(self, duration_ms: float, rate_max: float, rate_fn) -> list[float]:
        """
        Standard thinning algorithm for non-homogeneous Poisson processes.
        Generates candidates at rate_max, keeps each with probability
        rate_fn(t) / rate_max.
        """
        if rate_max <= 0:
            return []
        arrivals = []
        t = 0.0
        while True:
            inter_arrival = self.rng.exponential(1.0 / rate_max)
            t += inter_arrival
            if t >= duration_ms:
                break
            accept_prob = rate_fn(t) / rate_max
            if self.rng.random() < accept_prob:
                arrivals.append(t)
        return arrivals

    # ------------------------------------------------------------------
    # Task attribute generation
    # ------------------------------------------------------------------

    def _generate_task(self, arrival_time_ms: float, preset: dict) -> Task:
        task_id = self._next_task_id
        self._next_task_id += 1

        user_id = int(self.rng.integers(0, self.num_users))

        priority = self.rng.choice(
            list(preset["priority_weights"].keys()),
            p=list(preset["priority_weights"].values()),
        )

        cpu_requirement = float(self.rng.uniform(*preset["cpu_range"]))
        ram_requirement = float(self.rng.uniform(*preset["ram_range"]))
        data_size_mb = float(self.rng.uniform(*preset["data_size_range_mb"]))
        execution_time_ms = float(self.rng.uniform(*preset["exec_time_range_ms"]))

        slack_min, slack_max = SLACK_RANGES[priority]
        slack_factor = self.rng.uniform(slack_min, slack_max)
        deadline_ms = arrival_time_ms + execution_time_ms * slack_factor

        return Task(
            task_id=task_id,
            user_id=user_id,
            cpu_requirement=cpu_requirement,
            ram_requirement=ram_requirement,
            data_size_mb=data_size_mb,
            execution_time_ms=execution_time_ms,
            deadline_ms=deadline_ms,
            priority=priority,
            arrival_time_ms=arrival_time_ms,
        )