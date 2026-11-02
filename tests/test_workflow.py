import json
import numpy as np

from iov.config import Config, EnvironmentConfig, TrainingConfig
from iov.evaluation import episode, paired_difference
from iov.learner import Learner
from iov.replay import load_transitions
from iov.train import train


def test_training_writes_real_transitions_and_loadable_models(tmp_path):
    cfg = Config(EnvironmentConfig(agents=2, rsus=1, subtasks=2, horizon=4),
                 TrainingConfig(episodes=2, warmup_steps=8, imitation_steps=2,
                                batch_size=4, update_interval=2, eval_interval=1))
    run = train(cfg, tmp_path / "run")
    manifest = json.loads((run / "training_manifest.json").read_text())
    assert manifest["rl_steps"] == 8
    assert manifest["actor_updates_per_agent"] == 4
    assert manifest["test_seeds_used_in_training"] == []
    with np.load(run / "matching_data.npz", allow_pickle=False) as transitions:
        assert transitions["actions"].shape == (8, 2, 2)
        assert not np.array_equal(transitions["observations"], transitions["next_observations"])
        assert np.isfinite(transitions["rewards"]).all()
    restored = load_transitions(run / "matching_data.npz")
    assert len(restored) == 8
    assert restored[0].state.costs.shape == (2, 2, 5)
    assert restored[0].next_state.available.dtype == bool
    model = Learner.load(run / "best.pt")
    result = episode(cfg.environment, "trained", 300001, model, frames=True)
    assert len(result["frames"]) == 4
    assert result["metrics"]["attempted_subtasks"] > 0
    assert (run / "history.csv").read_text().count("\n") == 3


def test_paired_comparison_rejects_unpaired_scenarios():
    import pytest
    with pytest.raises(ValueError, match="same ordered"):
        paired_difference({"episodes": [{"seed": 1, "reward": 1}]},
                          {"episodes": [{"seed": 2, "reward": 1}]}, "reward")

