import itertools
import numpy as np
import pytest

from iov.config import EnvironmentConfig
from iov.matching import Matcher


def test_assignment_matches_exhaustive_optimum_and_respects_capacity():
    c = EnvironmentConfig(agents=2, rsus=1, subtasks=2, moving_slots=1, parked_slots=1, mec_slots=1)
    matcher = Matcher(c)
    scores = np.random.default_rng(42).normal(size=(2, 2, 5))
    rsu, available = np.zeros(2, dtype=int), np.ones((2, 5), dtype=bool)
    available[0, 1] = False
    best = -float("inf")
    for candidate in itertools.product(range(5), repeat=4):
        candidate = np.array(candidate).reshape(2, 2)
        try:
            matcher.validate(candidate, rsu, available)
        except ValueError:
            continue
        best = max(best, np.take_along_axis(scores, candidate[..., None], -1).sum())
    action, slots = matcher.solve(scores, rsu, available, return_slots=True)
    matcher.validate(action, rsu, available)
    assert len(set(slots.tolist())) == 4
    assert np.take_along_axis(scores, action[..., None], -1).sum() == pytest.approx(best)


def test_local_resources_are_private_and_remote_resources_are_regional():
    c = EnvironmentConfig(agents=2, rsus=2, subtasks=1)
    matcher = Matcher(c)
    allowed = matcher.allowed_edges(np.array([0, 1]), np.ones((2, 5), bool))
    assert allowed[0, 0] and not allowed[1, 0]
    assert allowed[1, 1] and not allowed[0, 1]
    for idx, (mode, owner) in enumerate(zip(matcher.mode, matcher.owner)):
        if mode in (1, 2, 3):
            assert allowed[owner, idx]
            assert not allowed[1 - owner, idx]


def test_infeasible_and_invalid_scores_fail_instead_of_returning_bad_edges():
    c = EnvironmentConfig(agents=2, rsus=1, subtasks=1)
    matcher = Matcher(c)
    with pytest.raises(ValueError, match="feasible"):
        matcher.solve(np.zeros((2, 1, 5)), np.zeros(2, int), np.zeros((2, 5), bool))
    with pytest.raises(ValueError, match="finite"):
        matcher.solve(np.full((2, 1, 5), np.nan), np.zeros(2, int), np.ones((2, 5), bool))

