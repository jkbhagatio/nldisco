"""Fast checks for the common feature-recovery selection contract."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from experiments.churchland.sweeps.s10_endpoint_v1.comparison_features import (
    EXPECTED_FEATURES,
    FeatureData,
    best_eligible,
    configuration_score,
    score_dense,
)


def test_orientation_threshold_and_trial_support():
    y = np.r_[np.ones(40, bool), np.zeros(40, bool)]
    x = np.column_stack((-y.astype(float), np.zeros(80)))
    feature = FeatureData(np.arange(80), y, np.arange(80), .5, {})
    first, constant = score_dense(x, feature)
    assert first['orientation'] == -1 and first['auroc'] == roc_auc_score(y, -x[:, 0])
    assert first['sel'] == 1 and first['achieved_rate'] == .5
    assert first['active_positive_trials'] == 40 and first['eligible']
    assert not constant['eligible'] and constant['achieved_rate'] == 0
    assert best_eligible([constant, first]) == first


def test_rows_and_ties_are_not_split():
    y = np.r_[np.ones(40, bool), np.zeros(40, bool)]
    x = np.r_[np.full(3, np.nan), np.zeros(40), np.ones(40)][:, None]
    feature = FeatureData(np.arange(3, 83), y, np.zeros(80, int), .1, {})
    row = score_dense(x, feature)[0]
    assert row['orientation'] == -1 and row['auroc'] == 1
    assert row['achieved_rate'] == 0 and not row['eligible']


def test_requires_four_features_and_equal_weight():
    winners = {name: {'auroc': value} for name, value in zip(EXPECTED_FEATURES, [.6, .7, .8, .9])}
    assert configuration_score(winners) == pytest.approx(.75)
    with pytest.raises(ValueError, match='four'):
        configuration_score({EXPECTED_FEATURES[0]: {'auroc': .9}})
    winners[EXPECTED_FEATURES[-1]] = None
    with pytest.raises(ValueError, match='eligible'):
        configuration_score(winners)
