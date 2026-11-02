from dataclasses import dataclass
import numpy as np

from .environment import State


@dataclass
class Transition:
    state: State
    action: np.ndarray
    reward: np.ndarray
    next_state: State
    terminated: bool


class ReplayBuffer:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.data = []
        self.cursor = 0

    def __len__(self):
        return len(self.data)

    def add(self, state, action, reward, next_state, terminated=False):
        item = Transition(state, action.copy(), reward.copy(), next_state, bool(terminated))
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.cursor] = item
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, count):
        if len(self.data) < count:
            raise ValueError("Replay buffer does not yet contain a complete batch")
        return [self.data[i] for i in self.rng.choice(len(self.data), count, replace=False)]


def save_transitions(path, transitions):
    """Save complete, pickle-free transitions for offline inspection and reuse."""
    arrays = {"schema_version": np.array(1),
              "actions": np.stack([x.action for x in transitions]),
              "rewards": np.stack([x.reward for x in transitions]),
              "terminated": np.array([x.terminated for x in transitions])}
    for prefix, attr in (("", "state"), ("next_", "next_state")):
        for name in ("observation", "adjacency", "costs", "rsu", "available"):
            key = "observations" if name == "observation" else name
            arrays[prefix + key] = np.stack([getattr(getattr(x, attr), name) for x in transitions])
    np.savez_compressed(path, **arrays)


def load_transitions(path):
    with np.load(path, allow_pickle=False) as arrays:
        if arrays["schema_version"].item() != 1:
            raise ValueError("Unsupported transition data schema")
        def state(prefix, i):
            return State(*(arrays[prefix + key][i].copy() for key in
                           ("observations", "adjacency", "costs", "rsu", "available")))
        return [Transition(state("", i), arrays["actions"][i].copy(), arrays["rewards"][i].copy(),
                           state("next_", i), bool(arrays["terminated"][i]))
                for i in range(len(arrays["actions"]))]

