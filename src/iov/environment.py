"""Seeded digital-twin simulator. All physical quantities use SI units.

Vehicles have bit queues; arrivals are Poisson, mobility is a periodic one-way
road, and remote processors expose reserved task slots per RSU. Predictions have
noise; rewards and queue service use the true state. See docs/METHODOLOGY.md.
"""
from dataclasses import dataclass
import numpy as np

from .config import EnvironmentConfig
from .matching import Matcher


@dataclass
class State:
    observation: np.ndarray
    adjacency: np.ndarray
    costs: np.ndarray
    rsu: np.ndarray
    available: np.ndarray


class OffloadingEnv:
    observation_size = 31  # paper's six scalar features + five twin features per mode

    def __init__(self, config=None):
        self.config = config or EnvironmentConfig()
        self.matcher = Matcher(self.config)
        self.unit_bits = self.config.task_mb * 8e6 / self.config.subtasks

    def reset(self, seed=0):
        c = self.config
        self.rng = np.random.default_rng(seed)
        self.time = 0
        self.positions = self.rng.uniform(0, c.road_m, c.agents)
        self.speeds = np.clip(self.rng.normal(22, 6, c.agents), 5, 38)
        self.local_capacity = c.local_ghz * self.rng.uniform(0.55, 1.25, c.agents)
        self.cycles = c.cycles_per_bit * self.rng.uniform(0.7, 1.5, c.agents)
        self.arrivals = self.rng.poisson(c.arrival_rate, c.agents) * self.unit_bits
        self.queues = self.arrivals.astype(float).copy()
        self._refresh()
        return self.state

    def _refresh(self):
        c = self.config
        n, m = c.agents, c.subtasks
        rsu = np.minimum((self.positions / (c.road_m / c.rsus)).astype(int), c.rsus - 1)
        distance = np.abs(self.positions[:, None] - self.positions)
        distance = np.minimum(distance, c.road_m - distance)
        adjacency = ((distance < 230) | (rsu[:, None] == rsu[None, :])).astype(np.float32)
        np.fill_diagonal(adjacency, 1)

        # Independent exogenous draws have fixed shapes, allowing paired evaluation
        # of different policies against exactly the same mobility/channel arrivals.
        self.cpu = np.stack([
            self.local_capacity / m,
            np.full(n, c.moving_ghz / c.moving_slots),
            np.full(n, c.parked_ghz / c.parked_slots),
            np.full(n, c.mec_ghz / c.mec_slots),
            np.full(n, 100.0),
        ], axis=1) * self.rng.uniform(0.75, 1.15, (n, 5))
        rsu_distance = np.abs(self.positions - (rsu + 0.5) * c.road_m / c.rsus)
        snr = self.rng.lognormal(2.0, 0.5, (n, 5)) / (1 + (rsu_distance[:, None] / 160) ** 2)
        rate = c.bandwidth_hz * np.log2(1 + snr)
        rate[:, 0] = np.inf
        available = self.rng.random((n, 5)) > np.array([0, 0.16, 0.08, 0.03, 0])
        available[:, [0, 4]] = True
        remaining_road = (rsu + 1) * c.road_m / c.rsus - self.positions
        contact = np.maximum(remaining_road / self.speeds, 0.01)
        self.contact = np.tile(contact[:, None], (1, 5))
        self.contact[:, [0, 4]] = np.inf
        self.available = available
        # Work is integer packets; zero-demand reserved slots perform no work.
        packet_bits = np.minimum(np.maximum(self.queues[:, None] - np.arange(m) * self.unit_bits, 0), self.unit_bits)
        self.demand = packet_bits
        cycles = packet_bits[:, :, None] * self.cycles[:, None, None]
        compute = cycles / (self.cpu[:, None, :] * 1e9)
        transmit = 2 * packet_bits[:, :, None] / rate[:, None, :]
        rtt = np.array([0.0, 0.01, 0.01, 0.02, 0.15])[None, None, :]
        self.latency = (compute + transmit + rtt) * (packet_bits[:, :, None] > 0)
        joules_per_gcycle = np.tile(np.array([0.8, 0.48, 0.32, 0.18, 0.12]), (n, 1))
        joules_per_gcycle[:, 0] *= (self.local_capacity / c.local_ghz) ** 2
        self.energy = cycles / 1e9 * joules_per_gcycle[:, None, :] + c.tx_power_w * transmit
        self.energy[:, :, 4] += 0.4 * packet_bits / 1e6  # cloud backhaul energy
        self.success = ((self.latency <= min(c.deadline_s, c.slot_s)) &
                        (self.latency <= self.contact[:, None, :]) & available[:, None, :])
        noise = np.clip(self.rng.normal(1, c.twin_relative_std, (n, 1, 5)), 0.5, 1.5)
        pred_latency = self.latency * noise
        pred_energy = self.energy * noise
        pred_success = ((pred_latency <= min(c.deadline_s, c.slot_s)) &
                        (pred_latency <= self.contact[:, None, :]) & available[:, None, :])
        q = self.queues / (m * self.unit_bits)
        served = pred_success * packet_bits[:, :, None] / self.unit_bits
        costs = (c.lyapunov_v * pred_energy / c.energy_scale_j - q[:, None, None] * served +
                 c.failure_penalty * (~pred_success) * (packet_bits[:, :, None] > 0))
        # A hypothetical full packet keeps twin features informative when idle.
        ref_cycles = self.unit_bits * self.cycles[:, None]
        ref_latency = ref_cycles / (self.cpu * 1e9) + 2 * self.unit_bits / rate + rtt.reshape(1, 5)
        ref_energy = ref_cycles / 1e9 * joules_per_gcycle + c.tx_power_w * 2 * self.unit_bits / rate
        ref_energy[:, 4] += 0.4 * self.unit_bits / 1e6
        features = np.stack([
            ref_energy * noise[:, 0] / c.energy_scale_j,
            ref_latency * noise[:, 0] / c.deadline_s,
            ((ref_latency * noise[:, 0] <= min(c.deadline_s, c.slot_s)) & (ref_latency * noise[:, 0] <= self.contact)).astype(float),
            available.astype(float),
            np.minimum(self.contact / c.deadline_s, 5) / 5,
        ], axis=-1)
        core = np.stack([self.positions / c.road_m, self.speeds / 40,
                         self.arrivals / (m * self.unit_bits), self.local_capacity / c.local_ghz,
                         np.full(n, c.deadline_s), q], axis=-1)
        observation = np.clip(np.concatenate([core, features.reshape(n, -1)], axis=-1), 0, 10).astype(np.float32)
        self.state = State(observation, adjacency, costs.astype(np.float32), rsu.copy(), available.copy())

    def matching_action(self):
        return self.matcher.solve(-self.state.costs, self.state.rsu, self.state.available)

    def step(self, actions):
        if self.time >= self.config.horizon:
            raise RuntimeError("Episode ended; call reset")
        c = self.config
        actions = np.asarray(actions)
        self.matcher.validate(actions, self.state.rsu, self.state.available)
        index = (np.arange(c.agents)[:, None], np.arange(c.subtasks)[None, :], actions)
        energy, latency, success = self.energy[index], self.latency[index], self.success[index]
        active = self.demand > 0
        completed = success & active
        served = (self.demand * completed).sum(axis=1)
        attempted = active.sum()
        prev_queues = self.queues.copy()
        new_arrivals = self.rng.poisson(c.arrival_rate, c.agents) * self.unit_bits
        self.queues = np.maximum(self.queues - served, 0) + new_arrivals
        # Eq. 39 with dimensionless energy and Eq. 40's per-agent feasibility bonus.
        team_energy = energy.sum() / (c.agents * c.energy_scale_j)
        feasible = np.all(success | ~active, axis=1)
        reward = np.exp(-team_energy) + np.where(feasible, 0.5, -0.5)
        info = {
            "energy_j": float(energy.sum()),
            "reward": float(reward.mean()),
            "queue_mbit": float(self.queues.mean() / 1e6),
            "queue_growth_mbit": float((self.queues - prev_queues).mean() / 1e6),
            "latency_s": float(latency[active].mean()) if attempted else 0.0,
            "attempted": int(attempted),
            "completed": int(completed.sum()),
            "served_mbit": float(served.sum() / 1e6),
            "arrivals_mbit": float(new_arrivals.sum() / 1e6),
            "deadline_success": float(completed.sum() / attempted) if attempted else 1.0,
            "mode_counts": [int(np.count_nonzero((actions == mode) & active)) for mode in range(5)],
            "positions": self.positions.tolist(),
            "queues_mbit": (self.queues / 1e6).tolist(),
            "actions": actions.tolist(),
            "adjacency": self.state.adjacency.astype(int).tolist(),
            "rsu": self.state.rsu.tolist(),
        }
        self.arrivals = new_arrivals
        self.positions = (self.positions + self.speeds * c.slot_s) % c.road_m
        self.speeds = np.clip(self.speeds + self.rng.normal(0, 0.5, c.agents), 5, 38)
        self.time += 1
        self._refresh()
        # Time-limit truncation is bootstrapped by the learner, not terminal failure.
        return self.state, reward.astype(np.float32), False, self.time >= c.horizon, info

