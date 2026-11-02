"""Exact maximum-weight, capacity-constrained matching for unit task slots.

Algorithm 1's per-vertex knapsack pseudocode does not resolve conflicting edges.
We expand each resource capacity into slots and solve the complete assignment.
This is an exact b-matching for equal-size task slots, not a general knapsack solver.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import EnvironmentConfig


class Matcher:
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        mode, owner = [], []
        for agent in range(config.agents):
            mode.extend([0] * config.subtasks)
            owner.extend([agent] * config.subtasks)
        for m, capacity in enumerate((config.moving_slots, config.parked_slots, config.mec_slots), 1):
            for rsu in range(config.rsus):
                mode.extend([m] * capacity)
                owner.extend([rsu] * capacity)
        mode.extend([4] * (config.agents * config.subtasks))
        owner.extend([-1] * (config.agents * config.subtasks))
        self.mode = np.asarray(mode)
        self.owner = np.asarray(owner)
        self.item_agent = np.repeat(np.arange(config.agents), config.subtasks)

    def allowed_edges(self, rsu, available):
        agents = self.item_agent[:, None]
        modes = self.mode[None, :]
        allowed = ((modes == 0) & (self.owner == agents)) | (modes == 4)
        allowed |= ((modes > 0) & (modes < 4) & (self.owner == np.asarray(rsu)[agents]))
        return allowed & np.asarray(available, dtype=bool)[agents, modes]

    def solve(self, scores, rsu, available, *, return_slots=False):
        scores = np.asarray(scores, dtype=np.float64)
        expected = (self.config.agents, self.config.subtasks, 5)
        if scores.shape != expected or not np.isfinite(scores).all():
            raise ValueError(f"Expected finite scores with shape {expected}")
        allowed = self.allowed_edges(rsu, available)
        utility = scores.reshape(-1, 5)[:, self.mode]
        costs = np.where(allowed, -utility, np.inf)
        try:
            row, col = linear_sum_assignment(costs)
        except ValueError as exc:
            raise ValueError("No feasible complete assignment exists") from exc
        if len(row) != len(self.item_agent) or not allowed[row, col].all():
            raise ValueError("No feasible complete assignment exists")
        result = np.empty(len(row), dtype=np.int64)
        result[row] = self.mode[col]
        modes = result.reshape(expected[:2])
        return (modes, col) if return_slots else modes

    def validate(self, actions, rsu, available):
        actions = np.asarray(actions)
        if actions.shape != (self.config.agents, self.config.subtasks):
            raise ValueError("One mode is required per agent and subtask")
        if not np.issubdtype(actions.dtype, np.integer) or np.any((actions < 0) | (actions > 4)):
            raise ValueError("Actions must be integer modes 0..4")
        if not np.asarray(available)[np.arange(self.config.agents)[:, None], actions].all():
            raise ValueError("An action uses an unavailable link")
        for mode, cap in enumerate((self.config.moving_slots, self.config.parked_slots, self.config.mec_slots), 1):
            for region in range(self.config.rsus):
                if np.count_nonzero(actions[np.asarray(rsu) == region] == mode) > cap:
                    raise ValueError("Shared resource capacity exceeded")

