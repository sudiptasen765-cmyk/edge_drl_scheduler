from environment.task_generator import TaskGenerator, WORKLOAD_PRESETS

gen = TaskGenerator(num_users=10, seed=42)

for workload in ["normal", "heavy", "burst", "variable", "unseen"]:
    tasks = gen.generate_episode(duration_ms=10000, workload_type=workload)
    print(f"\n=== {workload} ===")
    print(f"  Total tasks in 10s: {len(tasks)}")
    if tasks:
        print(f"  First task: {tasks[0].to_dict()}")
        priorities = [t.priority for t in tasks]
        print(f"  Priority mix: high={priorities.count('high')}, "
              f"medium={priorities.count('medium')}, low={priorities.count('low')}")
        # Check deadline > arrival for all tasks
        bad_deadlines = [t.task_id for t in tasks if t.deadline_ms <= t.arrival_time_ms]
        print(f"  Tasks with invalid deadlines: {len(bad_deadlines)} (should be 0)")
        # Check arrivals sorted
        arrival_times = [t.arrival_time_ms for t in tasks]
        print(f"  Sorted correctly: {arrival_times == sorted(arrival_times)}")
    gen.reset(seed=42)  # reset for fair comparison across workload types