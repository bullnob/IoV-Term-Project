"""Explicit units and defaults; undocumented paper choices are not hidden."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

MODES = ("Local", "Moving vehicle", "Parked vehicle", "MEC", "Cloud")


@dataclass
class EnvironmentConfig:
    agents: int = 6
    rsus: int = 2
    subtasks: int = 3
    horizon: int = 40
    road_m: float = 1000.0
    slot_s: float = 1.0
    deadline_s: float = 1.0
    task_mb: float = 0.6
    cycles_per_bit: float = 500.0
    arrival_rate: float = 2.3  # packets per vehicle per slot; each packet = task / M
    local_ghz: float = 2.5
    moving_ghz: float = 5.0
    parked_ghz: float = 8.0
    mec_ghz: float = 60.0
    moving_slots: int = 2
    parked_slots: int = 2
    mec_slots: int = 3
    tx_power_w: float = 0.1
    bandwidth_hz: float = 20e6
    twin_relative_std: float = 0.05
    lyapunov_v: float = 1.0
    energy_scale_j: float = 1.0
    failure_penalty: float = 3.0

    def __post_init__(self):
        for name in ("agents", "rsus", "subtasks", "horizon", "moving_slots", "parked_slots", "mec_slots"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("road_m", "slot_s", "deadline_s", "task_mb", "cycles_per_bit", "arrival_rate", "local_ghz", "moving_ghz", "parked_ghz", "mec_ghz", "tx_power_w", "bandwidth_hz", "energy_scale_j"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.twin_relative_std <= 0.5:
            raise ValueError("twin_relative_std must be in [0, 0.5]")


@dataclass
class TrainingConfig:
    seed: int = 7
    episodes: int = 100
    batch_size: int = 32
    replay_size: int = 1024
    update_interval: int = 10
    warmup_steps: int = 256
    imitation_steps: int = 160
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    gamma: float = 0.95
    tau: float = 0.01
    lyapunov_phi: float = 0.15
    noise_start: float = 0.8
    noise_end: float = 0.05
    eval_interval: int = 10
    hidden: tuple[int, int] = (80, 60)
    graph_hidden: int = 32
    threads: int = 1


@dataclass
class Config:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text())
        return cls(EnvironmentConfig(**raw.get("environment", {})), TrainingConfig(**raw.get("training", {})))

