from environment.edge_environment import EdgeEnvironment

server_configs = [
    {"server_id": 1, "cpu_cores": 4, "ram_gb": 8, "base_latency_ms": 5,
     "bandwidth_mbps": 100, "power_idle_w": 20, "power_max_w": 65,
     "power_exponent": 1.2, "max_queue_length": 20},
    {"server_id": 2, "cpu_cores": 8, "ram_gb": 16, "base_latency_ms": 12,
     "bandwidth_mbps": 200, "power_idle_w": 30, "power_max_w": 110,
     "power_exponent": 1.2, "max_queue_length": 30},
]

env = EdgeEnvironment(server_configs=server_configs, num_users=10, time_step_ms=10.0, seed=42)


def always_server_1(task, env):
    return 1


summary = env.run_episode(
    scheduler_fn=always_server_1,
    workload_type="normal",
    network_scenario="normal",
    duration_ms=5000.0,
    seed=42,
)

print("Episode summary:")
for k, v in summary.items():
    print(f"  {k}: {v}")

print("\nUser stats sample:")
for user_id, stats in list(env.user_stats.items())[:3]:
    print(f"  user {user_id}: {stats}")

print("\nFailed tasks sample:", env.failed_tasks[:3])