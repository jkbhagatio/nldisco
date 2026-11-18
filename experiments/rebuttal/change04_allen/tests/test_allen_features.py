"""Fast tests for the Allen stimulus alignment, split and association metrics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from experiments.rebuttal.change04_allen.data import PARAMS, bin_table, label_times, split_units
from experiments.rebuttal.change04_allen.features import association


def _pres(rows):
    df = pd.DataFrame(rows, columns=["start", "stop", "stim", "frame", "color"])
    for p in PARAMS:
        if p not in df:
            df[p] = np.nan
    return df


def test_label_times_inside_and_gap():
    starts, stops = np.array([1.0, 2.0]), np.array([1.5, 3.0])
    idx, inside = label_times(np.array([0.5, 1.2, 1.7, 2.9, 3.5]), starts, stops)
    np.testing.assert_array_equal(idx, [-1, 0, 0, 1, 1])
    np.testing.assert_array_equal(inside, [False, True, False, True, False])


def test_split_units_groups_movie_repeats():
    pres = _pres([(0, 1, "flashes", np.nan, 1), (1, 2, "flashes", np.nan, -1),
                  (2, 3, "natural_movie_one", 0, np.nan), (3, 4, "natural_movie_one", 1, np.nan),
                  (4, 5, "natural_movie_one", 0, np.nan), (5, 6, "natural_movie_one", 1, np.nan)])
    np.testing.assert_array_equal(split_units(pres), [0, 1, 2, 2, 3, 3])


def test_bin_table_delay_gray_and_split():
    pres = _pres([(0.1, 0.35, "flashes", np.nan, 1.0), (1.0, 1.25, "flashes", np.nan, -1.0)]
                 + [(2.0 + i, 2.25 + i, "flashes", np.nan, 1.0) for i in range(8)])
    df = bin_table(pres, np.array([[0.0, 0.05]]), n_bins=40, bin_s=0.05, delay_s=0.05,
                   train_fraction=0.8, seed=0)
    # Bin 3 ([0.15, 0.2)) sees the stimulus at 0.125 s: the first (white) flash.
    assert df.stim[3] == "flashes" and df.color[3] == 1.0
    # Bin 2 sees 0.075 s, before the first presentation; bin 9 sees 0.425 s, its gray gap.
    assert df.stim[2] == "none" and df.stim[9] == "flashes_gray" and np.isnan(df.color[9])
    assert df.pres_idx[9] == 0 and df.split_unit[9] == df.split_unit[3]
    assert df.invalid[0] and not df.invalid[1]
    # 10 presentations -> 2 held out; every bin of a unit shares its split.
    assert df[df.stim != "none"].groupby("split_unit").split.nunique().max() == 1
    units = df[df.stim != "none"].groupby("split_unit").split.first()
    assert (units == "test").sum() == 0 or (units == "test").sum() <= 2


def test_association_matches_definitions_and_sklearn():
    rng = np.random.default_rng(0)
    z = np.where(rng.random((500, 3)) < 0.3, rng.random((500, 3)), 0.0)
    pos = rng.random(500) < 0.4
    z[pos, 0] += 0.5  # latent 0 prefers the condition
    m = association(z, pos)
    tpr, fpr = (z[pos] > 0).mean(0), (z[~pos] > 0).mean(0)
    np.testing.assert_allclose(m["cov"], tpr)
    np.testing.assert_allclose(m["spec"], 1 - fpr)
    np.testing.assert_allclose(m["sel"], tpr / (tpr + fpr))
    np.testing.assert_allclose(m["auroc"], [roc_auc_score(pos, z[:, j]) for j in range(3)])
    assert m["auroc"][0] > 0.8


def test_association_requires_both_classes():
    with pytest.raises(ValueError):
        association(np.ones((4, 2)), np.ones(4, dtype=bool))


def test_selected_features_are_within_populations_and_scored_per_split():
    from experiments.rebuttal.change04_allen.results import FEATURES, score

    rng = np.random.default_rng(1)
    stims = ["gabors", "drifting_gratings", "drifting_gratings_gray", "natural_scenes",
             "natural_movie_three", "spontaneous"]
    n = 3000
    bins = pd.DataFrame({
        "stim": rng.choice(stims, n), "x_position": rng.choice([0.0, 10.0], n),
        "y_position": rng.choice([0.0, 10.0], n), "temporal_frequency": rng.choice([1.0, 8.0, 15.0], n),
        "frame": rng.integers(0, 3600, n).astype(float), "invalid": rng.random(n) < 0.05,
        "split": rng.choice(["train", "test"], n),
    })
    bins.loc[bins.stim == "natural_scenes", "frame"] = rng.choice([19.0, 3.0], (bins.stim == "natural_scenes").sum())
    bins.loc[bins.stim == "natural_movie_three", "frame"] = rng.choice(
        [115 * 30 + 5, 10.0], (bins.stim == "natural_movie_three").sum())
    for f in FEATURES:
        cond, pop = f.condition(bins), f.population(bins)
        assert cond.any() and not (cond & ~pop).any()
    acts = rng.random((n, 320)) * (rng.random((n, 320)) < 0.2)
    scores = score(acts, bins)
    assert len(scores) == 2 * len(FEATURES) and scores.auroc.between(0, 1).all()
