"""Tests for time-preserving boolean tick-track construction."""

import numpy as np
import pytest

from experiments.aeon.sweep_20260921.temporal_ticks import full_track


def test_tick_states_preserve_gaps_and_mask_invalid_positives():
    active, valid = full_track(np.array([2, 3, 5]), np.array([True, False, True]),
                               np.array([True, True, False]), 7)
    np.testing.assert_array_equal(np.flatnonzero(active), [2])
    np.testing.assert_array_equal(np.flatnonzero(valid), [2, 3])
    assert not valid[4]  # Missing endpoint is not a negative observation.


def test_tick_states_reject_reordered_or_duplicate_endpoints():
    for endpoints in ([2, 1], [2, 2], [-1, 2], [2, 7]):
        with pytest.raises(ValueError):
            full_track(np.array(endpoints), np.ones(2, dtype=bool), np.ones(2, dtype=bool), 7)


def test_empty_track_is_unavailable_not_observed_inactive():
    active, valid = full_track(np.array([], dtype=int), np.array([], dtype=bool),
                               np.array([], dtype=bool), 5)
    assert not active.any() and not valid.any()




