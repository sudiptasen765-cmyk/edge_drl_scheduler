from environment.edge_environment import EdgeEnvironment
from scheduling.fifo import FIFOScheduler
from scheduling.round_robin import RoundRobinScheduler
from scheduling.random_scheduler import RandomScheduler
from scheduling.greedy_scheduler import GreedyScheduler

server_configs = [
    {"server_id": 1, "cpu_cores": 4, "ram_gb": 8, "base_latency_ms": 5,
     "bandwidth_mbps": 100, "power_idle_w": 20, "power_max_w": 65,
     "power_exponent": 1.2, "max_queue_length": 20},
    {"server_id": 2, "cpu_cores": 8, "ram_gb": 16, "base_latency_ms": 12,
     "bandwidth_mbps": 200, "power_idle_w": 30, "power_max_w": 110,
     "power_exponent": 1.2, "max_queue_length": 30},
    {"server_id": 3, "cpu_cores": 2, "ram_gb": 4, "base_latency_ms": 2,
     "bandwidth_mbps": 50, "power_idle_w": 12, "power_max_w": 35,
     "power_exponent": 1.2, "max_queue_length": 15},
    {"server_id": 4, "cpu_cores": 12, "ram_gb": 32, "base_latency_ms": 18,
     "bandwidth_mbps": 300, "power_idle_w": 45, "power_max_w": 180,
     "power_exponent": 1.3, "max_queue_length": 40},
]

schedulers = {
    "FIFO": FIFOScheduler(),
    "Round Robin": RoundRobinScheduler(),
    "Random": RandomScheduler(seed=123),
    "Greedy": GreedyScheduler(),
}

print(f"{'Scheduler':<14} {'Completed':>10} {'Failed':>8} {'CompRate':>9} "
      f"{'AvgLatency':>11} {'DeadlineMet':>12} {'EnergyJ':>10}")
print("-" * 78)

for name, scheduler in schedulers.items():
    env = EdgeEnvironment(server_configs=server_configs, num_users=15, time_step_ms=10.0)
    if hasattr(scheduler, "reset"):
        scheduler.reset()
    summary = env.run_episode(
        scheduler_fn=scheduler,
        workload_type="normal",
        network_scenario="normal",
        duration_ms=10000.0,
        seed=42,  # SAME seed for every scheduler -> identical task stream
    )
    print(f"{name:<14} {summary['num_completed']:>10} {summary['num_failed']:>8} "
          f"{summary['completion_rate']:>9.2%} {summary['avg_latency_ms']:>11.1f} "
          f"{summary['deadline_met_rate']:>12.2%} {summary['total_energy_joules']:>10.1f}")