"""Position-conditioned attribution preserves equal sample weights and missing support."""

import numpy as np
import pytest

from experiments.synthetic.scripts.attribution_traceback import position_attribution_means


def test_position_means_include_boundaries_and_preserve_overall_attribution():
    contributions = np.array([[1., -2.], [5., 4.], [9., 6.], [13., 8.]])
    positions = np.array([0., .1, .5, 1.])
    means, counts = position_attribution_means(contributions, positions, np.linspace(0, 1, 5))
    np.testing.assert_array_equal(counts, [2, 0, 1, 1])
    np.testing.assert_allclose(means[[0, 2, 3]], [[3., 1.], [9., 6.], [13., 8.]])
    assert np.isnan(means[1]).all()
    np.testing.assert_allclose(
        np.nansum(means * counts[:, None], axis=0) / counts.sum(), contributions.mean(axis=0)
    )


@pytest.mark.parametrize("positions,edges", [
    ([1.1], [0., 1.]),
    ([-.1], [0., 1.]),
    ([np.nan], [0., 1.]),
    ([.5], [0., .5, .5, 1.]),
    ([.5], [0., np.nan, 1.]),
])
def test_position_means_reject_invalid_coordinates(positions, edges):
    with pytest.raises(ValueError):
        position_attribution_means(np.ones((1, 2)), np.array(positions), np.array(edges))
