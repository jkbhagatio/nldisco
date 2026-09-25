"""Synthetic checks for validation threshold selection and test scoring."""

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from experiments.decoder_metrics import (
    binary_metrics,
    select_balanced_threshold,
    trial_balanced_interval,
)


def test_threshold_matches_exhaustive_search_with_tied_scores():
    rng = np.random.default_rng(42)
    labels = rng.random(200) < .15
    scores = rng.integers(0, 12, 200).astype(float) / 12
    thresholds = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)[::-1]]
    expected = thresholds[np.argmax([balanced_accuracy_score(labels, scores >= t)
                                     for t in thresholds])]
    assert select_balanced_threshold(labels, scores) == expected


def test_threshold_can_be_small_for_rare_condition_and_is_frozen_for_test():
    validation_y = np.array([False, False, False, True, True])
    validation_p = np.array([.001, .002, .005, .01, .02])
    threshold = select_balanced_threshold(validation_y, validation_p)
    assert threshold == .01
    test_y = np.array([False, False, True, True])
    test_p = np.array([.001, .015, .005, .02])
    result = binary_metrics(test_y, test_p >= threshold)
    assert result == dict(tp=1, fn=1, tn=1, fp=1, sensitivity=.5,
                          specificity=.5, balanced_accuracy=.5)
    assert result["balanced_accuracy"] == balanced_accuracy_score(test_y, test_p >= threshold)


def test_objective_ties_choose_highest_threshold_without_splitting_ties():
    labels = np.array([True, False, True, False])
    assert select_balanced_threshold(labels, np.array([.9, .8, .7, .6])) == .9
    scores = np.full(4, .1)
    assert select_balanced_threshold(labels, scores) > .1


@pytest.mark.parametrize("scores", [[np.nan, .1], [np.inf, .1], [-.1, .1], [.1, 1.1]])
def test_invalid_probabilities(scores):
    with pytest.raises(ValueError):
        select_balanced_threshold(np.array([True, False]), np.array(scores))


@pytest.mark.parametrize("labels", [[], [True, True], [False, False]])
def test_requires_both_classes(labels):
    y = np.array(labels, dtype=bool)
    with pytest.raises(ValueError):
        select_balanced_threshold(y, np.zeros(len(y)))
    with pytest.raises(ValueError):
        binary_metrics(y, y)


def test_bootstrap_matches_explicit_whole_trial_resampling():
    trials = np.repeat(np.arange(4), [3, 4, 2, 5])
    y = np.array([True, False] * 7)
    predicted = np.array([True, True, False, False, False, True, True] * 2)
    rng = np.random.default_rng(734)
    values = []
    for draw in rng.integers(4, size=(300, 4)):
        rows = np.concatenate([np.flatnonzero(trials == trial) for trial in draw])
        values.append(balanced_accuracy_score(y[rows], predicted[rows]))
    result = trial_balanced_interval(y, predicted, trials)
    np.testing.assert_allclose([result["ci_lower"], result["ci_upper"]],
                               np.quantile(values, [.025, .975]))
    assert result["bootstrap_valid_replicates"] == 300
