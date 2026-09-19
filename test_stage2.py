from environment.edge_server import EdgeServer

# Create one server matching Server 1 from server_config.yaml
server = EdgeServer(
    server_id=1, cpu_cores=4, ram_gb=8,
    base_latency_ms=5, bandwidth_mbps=100,
    power_idle_w=20, power_max_w=65, power_exponent=1.2,
    max_queue_length=20,
)

print("Idle power (0% util):", server.current_power_draw(), "W")

# Assign a task using 2 of 4 cores, 2GB RAM, 500ms execution time
accepted = server.assign_task(
    task_id=1, user_id=100, cpu_requirement=2, ram_requirement=2,
    execution_time_ms=500, deadline_ms=1000, current_time_ms=0,
)
print("Task accepted:", accepted)

# Step forward 100ms - task should move from queue to running
summary = server.step(time_delta_ms=100, current_time_ms=100)
print("After 100ms:", server)
print("Step summary:", summary)

# Step forward until task finishes (needs 500ms total, already did 100ms)
for i in range(4):
    summary = server.step(time_delta_ms=100, current_time_ms=200 + i * 100)
    print(f"t={200 + i * 100}ms:", server, "finished:", summary["finished_task_ids"])

print("Total energy consumed (J):", round(server.total_energy_joules, 2))
print("Completed tasks:", server.completed_task_count)