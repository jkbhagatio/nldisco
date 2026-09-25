"""Fast checks of S10 label geometry and protected probe tuning."""
import numpy as np
import pytest

from .feature_decoding import label_windows, trial_interval, tune_probe


def test_fixed_label_geometry():
    speed = np.full((5, 10), 600.)
    speed[0] = np.linspace(200., 0., 10)
    vx, vy = np.full((5, 10), -400.), np.zeros((5, 10))
    tx, ty = np.array([-144., 125., 125., -144., 0.]), np.array([-52., -18., -18., -52., 0.])
    labels = label_windows(speed, vx, vy, tx, ty, np.array([.1, .3, 1.5, .2, .1]),
                            np.zeros(5), np.ones(5))
    assert labels['recent_braking'][1].tolist() == [True, False, False, False, False]
    assert labels['early_leftward'][1].tolist() == [True, False, False, False, True]
    assert labels['fast_target_specific'][1].tolist() == [False, False, False, True, False]
    assert labels['late_target_specific'][1].tolist() == [False, True, False, False, False]
    speed[1, 0] = np.nan
    changed = label_windows(speed, vx, vy, tx, ty, np.zeros(5), np.zeros(5), np.ones(5))
    assert not changed['recent_braking'][0][1]
    assert changed['late_target_specific'][0][1], 'Late target does not require available kinematics.'


def test_tuning_scaler_and_test_isolation():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(300, 3))
    feature = dict(rows=np.arange(200), labels=values[:200, 0] > 0,
                   trials=np.repeat(np.arange(20), 10), split=np.repeat([0, 1], [160, 40]))
    (scaler, model), first = tune_probe(values, feature, (.01, .1))
    changed = values.copy()
    changed[200:] += 10000
    _, second = tune_probe(changed, feature, (.01, .1))
    assert first == second
    np.testing.assert_allclose(scaler.mean_, values[:160].mean(0))
    assert model.class_weight is None and model.fit_intercept
    bad = dict(feature, split=feature['split'].copy())
    bad['split'][-1] = 2
    with pytest.raises(ValueError, match='development'):
        tune_probe(values, bad, (.1,))


def test_trial_bootstrap():
    y = np.tile([False, True], 30)
    trials = np.repeat(np.arange(10), 6)
    low, high, valid = trial_interval(y, y.astype(float), trials, 30, 42)
    assert (low, high, valid) == (1., 1., 30)
