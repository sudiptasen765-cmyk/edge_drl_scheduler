from environment.edge_server import EdgeServer
from environment.network import Network

server = EdgeServer(
    server_id=1, cpu_cores=4, ram_gb=8,
    base_latency_ms=5, bandwidth_mbps=100,
    power_idle_w=20, power_max_w=65, power_exponent=1.2,
    max_queue_length=20,
)

net = Network(congestion_latency_per_transfer_ms=2.0)

# --- Test 1: single transfer, normal scenario ---
net.set_scenario("normal")
net.start_transfer(server.server_id)
print("Single transfer:")
print("  Effective latency:", net.effective_latency_ms(server), "ms")
print("  Effective bandwidth:", net.effective_bandwidth_mbps(server), "Mbps")
print("  Transfer time for 10MB:", net.transfer_time_ms(10, server), "ms")

# --- Test 2: three concurrent transfers, equal split bandwidth ---
net.start_transfer(server.server_id)
net.start_transfer(server.server_id)
print("\nThree concurrent transfers:")
print("  Active transfers:", net.get_active_transfers(server.server_id))
print("  Effective latency:", net.effective_latency_ms(server), "ms")
print("  Effective bandwidth:", net.effective_bandwidth_mbps(server), "Mbps (should be ~1/3 of 100)")

# --- Test 3: switch to congested scenario ---
net.set_scenario("congested")
print("\nSame 3 transfers, but 'congested' scenario:")
print("  Effective latency:", net.effective_latency_ms(server), "ms (should be much higher)")
print("  Effective bandwidth:", net.effective_bandwidth_mbps(server), "Mbps (should be lower)")

# --- Test 4: end transfers, back to normal ---
net.end_transfer(server.server_id)
net.end_transfer(server.server_id)
net.end_transfer(server.server_id)
net.set_scenario("normal")
print("\nAfter ending all transfers:")
print("  Active transfers:", net.get_active_transfers(server.server_id))
print("  Effective bandwidth:", net.effective_bandwidth_mbps(server), "Mbps (should be back to 100)")