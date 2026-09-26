"""Consistent latent identities and example populations across appendix figures."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.synthetic.scripts import appendix, paper


def _records():
    return {direction: {"summary": {"seed": seed, "selected_directions": {
        direction: {"latent_idx": feature}}}, "artifacts": {
            f"{direction}_feature": np.asarray(feature),
            f"{direction}_clean_trajectories": np.array([[0., .8], [.1, .7]])}}
        for direction, seed, feature in (("right", 0, 20), ("left", 2, 66))}


def test_temporal_figures_use_same_pinned_features_and_distinct_pools(monkeypatch):
    records = _records()
    monkeypatch.setattr(paper, "choose_temporal", lambda results, arch, direction: records[direction])
    monkeypatch.setattr(paper, "_save", lambda *_: None)
    calls = []
    monkeypatch.setattr(paper, "_decoder", lambda *args: None)
    monkeypatch.setattr(paper, "_trajectories", lambda ax, record, direction,
                        ramp_conditioned=False: calls.append((direction, record, ramp_conditioned)))
    fig = appendix.temporal_features_figure({})
    plt.close(fig)
    fig = appendix.clean_crossings_figure({})
    assert fig.axes[0].get_ylim()[0] <= 0
    assert fig.axes[0].get_ylim()[1] >= .8
    assert [(d, clean) for d, _, clean in calls] == [
        ("right", False), ("left", False), ("right", True), ("left", True)]
    assert calls[0][1] is calls[2][1] and calls[1][1] is calls[3][1]
    plt.close(fig)
    records["right"]["summary"]["seed"] = 1
    with pytest.raises(ValueError, match="examples have changed"):
        appendix.selected_features({})
