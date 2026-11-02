import numpy as np
import pytest
import torch

from iov.config import Config, EnvironmentConfig, TrainingConfig
from iov.environment import OffloadingEnv
from iov.learner import Learner, bellman_target
from iov.networks import GraphActor, soft_update
from iov.replay import ReplayBuffer


def small_setup():
    config = Config(EnvironmentConfig(agents=2, rsus=1, subtasks=2, horizon=8), TrainingConfig(batch_size=4))
    learner, env = Learner(config), OffloadingEnv(config.environment)
    buffer = ReplayBuffer(6, 7)
    state = env.reset(1)
    for _ in range(8):
        action = env.matching_action()
        nxt, reward, terminal, _, _ = env.step(action)
        buffer.add(state, action, reward, nxt, terminal)
        state = nxt
    return config, learner, buffer, state


def test_actor_uses_neighbors_but_not_disconnected_nodes():
    torch.manual_seed(21)
    actor = GraphActor(31, 2, agent=0)
    obs = torch.randn(1, 3, 31)
    graph = torch.eye(3)[None]
    graph[0, 0, 1] = 1
    expected = actor(obs, graph)
    changed = obs.clone()
    changed[:, 2] += 100
    torch.testing.assert_close(actor(changed, graph), expected)
    changed[:, 1] += 100
    assert not torch.allclose(actor(changed, graph), expected)


def test_rl_updates_all_agents_and_targets_without_target_gradients():
    _, learner, buffer, _ = small_setup()
    actor_before = [torch.cat([p.detach().flatten().clone() for p in a.parameters()]) for a in learner.actors]
    critic_before = [torch.cat([p.detach().flatten().clone() for p in q.parameters()]) for q in learner.critics]
    target_before = [p.clone() for p in learner.target_actors.parameters()]
    metrics = learner.update(buffer.sample(4))
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["actor_gradient_norm"] > 0
    for k in range(2):
        assert not torch.equal(actor_before[k], torch.cat([p.detach().flatten() for p in learner.actors[k].parameters()]))
        assert not torch.equal(critic_before[k], torch.cat([p.detach().flatten() for p in learner.critics[k].parameters()]))
    assert any(not torch.equal(before, after) for before, after in zip(target_before, learner.target_actors.parameters()))
    assert all(p.grad is None for p in learner.target_actors.parameters())
    assert all(p.grad is None for p in learner.target_critics.parameters())


def test_bellman_mask_and_exact_soft_target_update():
    reward = torch.tensor([1., 1.])
    result = bellman_target(reward, torch.tensor([1., 0.]), torch.tensor([10., 10.]), 0.9)
    torch.testing.assert_close(result, torch.tensor([1., 10.]))
    source, target = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    with torch.no_grad():
        for p in source.parameters():
            p.fill_(2)
        for p in target.parameters():
            p.fill_(0)
    soft_update(target, source, 0.25)
    for p in target.parameters():
        torch.testing.assert_close(p, torch.full_like(p, 0.5))


def test_checkpoint_roundtrip_and_replay_capacity(tmp_path):
    _, learner, replay, state = small_setup()
    assert len(replay) == 6
    with pytest.raises(ValueError):
        replay.sample(7)
    learner.update(replay.sample(4))
    expected = learner.act(state)
    path = tmp_path / "checkpoint.pt"
    learner.save(path)
    restored = Learner.load(path)
    np.testing.assert_array_equal(restored.act(state), expected)
    assert restored.updates == 1

