import numpy as np
import pytest

from iov.config import EnvironmentConfig
from iov.environment import OffloadingEnv


def test_queue_conservation_and_unclipped_backlog():
    env = OffloadingEnv(EnvironmentConfig(arrival_rate=12, horizon=8))
    env.reset(44)
    for _ in range(8):
        before = env.queues.sum()
        state, reward, terminated, truncated, info = env.step(env.matching_action())
        assert env.queues.sum() == pytest.approx(before - info["served_mbit"] * 1e6 + info["arrivals_mbit"] * 1e6)
        assert np.isfinite(state.observation).all() and np.isfinite(reward).all()
        assert np.all(env.queues >= 0)
        assert not terminated
    assert truncated
    assert env.queues.mean() > 10 * env.unit_bits  # no silent queue clipping or dropped workload
    with pytest.raises(RuntimeError, match="reset"):
        env.step(env.matching_action())


def test_same_seed_gives_same_exogenous_traffic_across_policies():
    a, b = OffloadingEnv(), OffloadingEnv()
    a.reset(9)
    b.reset(9)
    for _ in range(12):
        a.step(a.matching_action())
        b.step(np.zeros((6, 3), dtype=np.int64))
        np.testing.assert_array_equal(a.positions, b.positions)
        np.testing.assert_array_equal(a.arrivals, b.arrivals)
        np.testing.assert_array_equal(a.cpu, b.cpu)
        np.testing.assert_array_equal(a.available, b.available)


def test_deadline_failures_retain_packets_and_count_consumed_energy():
    env = OffloadingEnv(EnvironmentConfig(deadline_s=1e-8, horizon=1))
    env.reset(12)
    old = env.queues.copy()
    _, _, _, _, info = env.step(np.zeros((6, 3), dtype=np.int64))
    assert info["completed"] == 0
    assert info["energy_j"] > 0
    np.testing.assert_allclose(env.queues, old + env.arrivals)


def test_matching_rollout_is_feasible_and_graph_has_self_loops():
    env = OffloadingEnv(EnvironmentConfig(horizon=15))
    state = env.reset(21)
    for _ in range(15):
        assert state.observation.shape == (6, 31)
        assert np.diag(state.adjacency).all()
        np.testing.assert_array_equal(state.adjacency, state.adjacency.T)
        state, _, _, _, _ = env.step(env.matching_action())

