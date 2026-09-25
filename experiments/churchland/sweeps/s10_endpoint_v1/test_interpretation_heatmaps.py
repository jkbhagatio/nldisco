"""Support and temporal alignment checks for interpretation heatmaps."""

import numpy as np

from experiments.churchland.sweeps.s10_endpoint_v1.interpretation_heatmaps import (
    activity_grid,
    slope_history,
)


def test_slopes_use_five_bins_inside_the_input():
    times = np.arange(10) * .05
    speed = np.stack((100 + 4 * times, 100 - 7 * times))
    lags, slopes = slope_history(speed)
    np.testing.assert_allclose(lags, [-.25, -.2, -.15, -.1, -.05, 0])
    np.testing.assert_allclose(slopes, np.repeat([[4.], [-7.]], 6, axis=1))
    altered = speed.copy()
    altered[:, -1] += 10
    _, updated = slope_history(altered)
    np.testing.assert_allclose(updated[:, :-1], slopes[:, :-1])
    assert np.all(updated[:, -1] > slopes[:, -1])


def test_distinct_trial_support_and_final_edge_are_consistent():
    x = np.r_[np.zeros(20), np.full(20, 2.)]
    y = np.zeros(40)
    trials = np.r_[np.zeros(20), np.repeat(np.arange(10), 2)]
    active = np.tile([True, False], 20)
    rate, counts, support = activity_grid(x, y, active, trials,
                                         np.array([0., 1., 2.]), np.array([0., 1.]))
    np.testing.assert_array_equal(counts, [[20, 20]])
    np.testing.assert_array_equal(support, [[1, 10]])
    assert np.isnan(rate[0, 0])
    assert rate[0, 1] == .5


def test_missing_and_out_of_range_observations_do_not_add_support():
    rate, counts, support = activity_grid(
        np.array([np.nan, -1., 3.]), np.zeros(3), np.ones(3, dtype=bool),
        np.arange(3), np.array([0., 2.]), np.array([0., 1.]))
    np.testing.assert_array_equal(counts, [[0]])
    np.testing.assert_array_equal(support, [[0]])
    assert np.isnan(rate).all()
