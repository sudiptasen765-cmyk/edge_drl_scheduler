"""
network.py

Models the network layer between users/IoT devices and edge servers.

Two independent effects are modeled:

1. Congestion latency:
   Each active data transfer to a server adds a small extra delay to every
   OTHER transfer on that same server, on top of the server's fixed
   base_latency_ms. This models contention on the link/switch serving that
   server - more simultaneous transfers, more queuing delay at the network
   layer (separate from the server's own CPU queue).

       effective_latency_ms = (base_latency_ms + active_transfers * congestion_latency_per_transfer_ms)
                               * latency_multiplier

2. Bandwidth sharing (equal split):
   A server's bandwidth_mbps is its TOTAL capacity. When multiple transfers
   are active at once, each gets an equal share - this mirrors how shared
   network links behave under contention (roughly how TCP flows fair-share
   a link in practice).

       effective_bandwidth_mbps = (server.bandwidth_mbps * bandwidth_multiplier)
                                   / max(1, active_transfers)

The `latency_multiplier` / `bandwidth_multiplier` fields are global scenario
controls - NOT per-server. They exist so experiments (e.g. "network
congestion" or "unseen conditions") can degrade the ENTIRE network's
conditions uniformly without having to edit every server's static config.
This is the lever Experiment 4 and the generalization test use.
"""

from dataclasses import dataclass, field


@dataclass
class NetworkScenario:
    """
    A named, reusable network condition preset. Experiments select one of
    these instead of hand-tuning multipliers inline, so results are
    comparable and the scenario used is always explicit and loggable.
    """
    name: str
    latency_multiplier: float = 1.0
    bandwidth_multiplier: float = 1.0


# Common presets, used across experiments so "congestion" always means the
# same thing everywhere it's referenced.
NETWORK_SCENARIOS = {
    "normal": NetworkScenario("normal", latency_multiplier=1.0, bandwidth_multiplier=1.0),
    "congested": NetworkScenario("congested", latency_multiplier=3.0, bandwidth_multiplier=0.4),
    "unseen": NetworkScenario("unseen", latency_multiplier=2.2, bandwidth_multiplier=0.6),
}


class Network:
    """
    Tracks active data transfers per server and computes effective latency
    and transfer time. This is a shared object passed into the simulation
    environment - servers themselves do not know about network contention,
    they only know their own static base_latency_ms / bandwidth_mbps.
    """

    def __init__(self, congestion_latency_per_transfer_ms: float = 2.0):
        self.congestion_latency_per_transfer_ms = congestion_latency_per_transfer_ms

        # server_id -> number of currently active (in-flight) transfers
        self.active_transfers: dict[int, int] = {}

        # Global scenario multipliers - see NetworkScenario above.
        self.latency_multiplier: float = 1.0
        self.bandwidth_multiplier: float = 1.0
        self.current_scenario_name: str = "normal"

    # ------------------------------------------------------------------
    # Scenario control (used by experiments to simulate different
    # network conditions without touching server configs)
    # ------------------------------------------------------------------

    def set_scenario(self, scenario_name: str):
        """
        Switch the whole network to a named scenario (see NETWORK_SCENARIOS).
        Used by experiments/congestion.py and experiments/unseen.py.
        """
        if scenario_name not in NETWORK_SCENARIOS:
            raise ValueError(
                f"Unknown network scenario '{scenario_name}'. "
                f"Available: {list(NETWORK_SCENARIOS.keys())}"
            )
        scenario = NETWORK_SCENARIOS[scenario_name]
        self.latency_multiplier = scenario.latency_multiplier
        self.bandwidth_multiplier = scenario.bandwidth_multiplier
        self.current_scenario_name = scenario.name

    # ------------------------------------------------------------------
    # Transfer lifecycle - call these when a task's data transfer
    # starts and ends, so contention is tracked accurately.
    # ------------------------------------------------------------------

    def start_transfer(self, server_id: int):
        self.active_transfers[server_id] = self.active_transfers.get(server_id, 0) + 1

    def end_transfer(self, server_id: int):
        if server_id in self.active_transfers:
            self.active_transfers[server_id] = max(0, self.active_transfers[server_id] - 1)

    def get_active_transfers(self, server_id: int) -> int:
        return self.active_transfers.get(server_id, 0)

    # ------------------------------------------------------------------
    # Effective latency / bandwidth / transfer time calculations
    # ------------------------------------------------------------------

    def effective_latency_ms(self, server) -> float:
        """
        server: an EdgeServer instance (reads its base_latency_ms).
        Adds congestion delay proportional to how many transfers are
        currently active on this server, then applies the scenario multiplier.
        """
        active = self.get_active_transfers(server.server_id)
        congestion_delay = active * self.congestion_latency_per_transfer_ms
        return (server.base_latency_ms + congestion_delay) * self.latency_multiplier

    def effective_bandwidth_mbps(self, server) -> float:
        """
        Equal-split bandwidth sharing: total capacity divided among all
        currently active transfers on this server (including the one being
        queried - i.e. call this AFTER start_transfer() has been called for
        the transfer you're timing, so it counts itself).
        """
        active = max(1, self.get_active_transfers(server.server_id))
        return (server.bandwidth_mbps * self.bandwidth_multiplier) / active

    def transfer_time_ms(self, data_size_mb: float, server) -> float:
        """
        Time to move data_size_mb megabytes to/from this server, given
        current effective bandwidth.

        Unit conversion: bandwidth is in Mbps (megabits/sec), data size is
        in MB (megabytes). 1 MB = 8 Mb.
        """
        bandwidth_mbps = self.effective_bandwidth_mbps(server)
        if bandwidth_mbps <= 0:
            return float("inf")
        data_size_megabits = data_size_mb * 8
        time_seconds = data_size_megabits / bandwidth_mbps
        return time_seconds * 1000.0

    def total_network_delay_ms(self, data_size_mb: float, server) -> float:
        """
        Convenience method combining latency + transfer time - the total
        network-caused delay a task experiences reaching/leaving this server.
        This is what the environment adds to a task's effective completion time.
        """
        return self.effective_latency_ms(server) + self.transfer_time_ms(data_size_mb, server)

    # ------------------------------------------------------------------
    # Reset / state export
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all dynamic state. Does NOT reset the scenario - call
        set_scenario('normal') explicitly if you want that too."""
        self.active_transfers = {}

    def get_state(self, server_id: int) -> dict:
        return {
            "server_id": server_id,
            "active_transfers": self.get_active_transfers(server_id),
            "latency_multiplier": self.latency_multiplier,
            "bandwidth_multiplier": self.bandwidth_multiplier,
            "scenario": self.current_scenario_name,
        }

    def __repr__(self):
        return (
            f"Network(scenario={self.current_scenario_name}, "
            f"active_transfers={dict(self.active_transfers)})"
        )