"""Fast checks of the split, behavior clocks, rule language, scoring and features."""
import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from experiments.rebuttal.change03_discovery import data, features, rules, scoring


def test_split_fractions_and_no_shared_bins():
    n = 20_000
    part, segment = data.bin_parts(n)
    assert (part == data.HELDOUT).mean() == 0.25
    assert np.isclose((part == data.VALIDATION).mean(), 0.075)
    starts = np.arange(n - data.SEQ_LEN + 1)
    window = data.window_parts(starts, n)
    assert (window == data.DROPPED).sum() == 19 * (len(np.unique(segment)) - 1)
    bins = starts[:, None] + np.arange(data.SEQ_LEN)
    held = np.unique(bins[window == data.HELDOUT])
    discovery = np.unique(bins[(window == data.FIT) | (window == data.VALIDATION)])
    assert not np.intersect1d(held, discovery).size
    assert (part[held] == data.HELDOUT).all() and (part[discovery] != data.HELDOUT).all()
    fit = np.unique(bins[window == data.FIT])
    assert (part[fit] == data.FIT).all()


def test_lagged_pairs_stay_inside_a_portion():
    starts = np.array([0, 1, 2, 5, 6, 7, 8])
    member = np.array([True, True, True, True, True, False, True])
    assert data.lagged_pairs(starts, member, 1).tolist() == [0, 1, 3]
    assert data.lagged_pairs(starts, member, 2).tolist() == [0, 4]


def test_support_clocks_and_wheel_bouts():
    assert data.complete_support(np.array([1, 1, 0, 1, 1], bool), 2).tolist() == \
        [False, True, False, False, True]
    since, until = data.event_clocks(np.array([2, 5]), 8)
    np.testing.assert_allclose(since[2:], np.array([0, 1, 2, 0, 1, 2]) * data.BIN_SECONDS)
    assert np.isnan(since[:2]).all() and np.isnan(until[6:]).all()
    np.testing.assert_allclose(until[:6], np.array([2, 1, 0, 2, 1, 0]) * data.BIN_SECONDS)
    wheel = np.zeros(200)
    wheel[20:60] = 1.   # one 0.8 s bout with a 0.1 s pause
    wheel[35:40] = 0.
    wheel[100:110] = 1.  # 0.2 s blip: removed
    onsets, offsets = data.wheel_bouts(wheel, np.ones(200, bool))
    assert onsets.tolist() == [20] and offsets.tolist() == [60]


def test_rules_evaluate_and_reject():
    variables = {name: np.zeros(4) for name in data.VARIABLES}
    variables.update(speed=np.array([0., 30., 150., 150.]),
                     heading=np.array([np.nan, 10., 170., -170.]),
                     wheel_speed=np.array([0., 0., .3, .3]), x=np.array([0., 3., 0., 0.]),
                     y=np.array([0., 4., 0., 0.]))
    support = dict(camera=np.array([True, True, True, False]), wheel=np.ones(4, bool),
                   all=np.ones(4, bool))
    mask, valid = rules.evaluate("abs(heading) >= 135 and speed >= 100", variables, support)
    assert mask.tolist() == [False, False, True, False] and valid.tolist() == [1, 1, 1, 0]
    mask, valid = rules.evaluate("wheel_speed > 0.05", variables, support)
    assert mask.tolist() == [False, False, True, True] and valid.all()
    assert rules.evaluate("not (heading > 0)", variables, support)[0].tolist() == [1, 0, 0, 0]
    assert rules.evaluate("hypot(x, y) == 5 or 20 <= speed < 100", variables, support)[0].tolist() \
        == [0, 1, 0, 0]
    assert rules.names_used("speed > 5 and area < 3") == {"speed", "area"}
    for bad in ("speed", "speed > ", "__import__('os')", "latent > 3", "speed > 5 and area",
                "1 < 2", "speed.real > 1", "(speed > 1) + 1 > 0", "max(speed, 1) > 0"):
        with pytest.raises(rules.RuleError):
            rules.evaluate(bad, variables, support)


def test_auroc_matrix_matches_sklearn_with_ties_and_populations():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(500, 4)).astype(np.float32)
    values[:, 1] = np.where(values[:, 1] > 1, values[:, 1], 0)  # sparse with many ties
    values[:, 2] = np.round(values[:, 2])
    labels = rng.random((500, 3)) < 0.3
    valid = rng.random((500, 3)) < 0.8
    result = scoring.auroc_matrix(torch.as_tensor(values), torch.as_tensor(labels & valid),
                                  torch.as_tensor(~labels & valid)).numpy()
    for i in range(4):
        for j in range(3):
            expected = roc_auc_score(labels[valid[:, j], j], values[valid[:, j], i])
            assert np.isclose(result[i, j], expected)
            assert np.isclose(scoring.auroc(values[valid[:, j], i].astype(float),
                                            labels[valid[:, j], j]), expected)


def test_scoring_rule_passes_planted_and_fails_noise():
    rng = np.random.default_rng(1)
    n = 40_000
    mask = rng.random(n) < 0.1
    valid = np.ones(n, bool)
    discovery = np.arange(n) % 4 != 3
    sparse = np.where(mask & (rng.random(n) < 0.8), rng.random(n) + 0.1, 0.)
    sparse[~mask & (rng.random(n) < 0.02)] = 0.2
    result = scoring.evaluate(sparse, 1, mask, valid, discovery, ~discovery)
    assert result["passed"] and result["threshold"] > 0 and result["sel"] > 0.9
    assert np.isclose(result["cov"], 0.8, atol=0.05) and result["spec"] > 0.95
    dense = -(mask * 2. + rng.normal(size=n))  # low when the condition holds
    assert scoring.evaluate(dense, -1, mask, valid, discovery, ~discovery)["passed"]
    assert not scoring.evaluate(dense, 1, mask, valid, discovery, ~discovery)["passed"]
    noise = scoring.evaluate(rng.normal(size=n), 1, mask, valid, discovery, ~discovery)
    assert not noise["passed"] and abs(noise["auroc"] - 0.5) < 0.05
    rare = np.zeros(n, bool)
    rare[:100] = True
    assert not scoring.evaluate(sparse, 1, rare, valid, discovery, ~discovery)["scorable"]
    assert scoring.youden_threshold(np.array([0., 0., 1., 2.]), np.array([0, 0, 1, 1], bool)) == 1.


def synthetic_behavior(n: int = 6000, seed: int = 0):
    rng = np.random.default_rng(seed)
    v = {name: rng.normal(size=n) for name in data.VARIABLES}
    v.update(x=rng.uniform(200, 1200, n), y=rng.uniform(100, 1000, n),
             speed=rng.exponential(30, n), area=rng.uniform(300, 700, n),
             wheel_speed=rng.exponential(.03, n), acceleration=rng.normal(0, 200, n),
             time_since_pellet=rng.uniform(0, 50, n), time_to_pellet=rng.uniform(0, 50, n))
    v["x"][: n // 3], v["y"][: n // 3] = 410., 230.
    v["patch_distance"] = np.hypot(v["x"] - 890, v["y"] - 850)
    v["heading"] = np.where(v["speed"] >= 20, rng.uniform(-180, 180, n), np.nan)
    v["heading_error_to_patch"] = np.where(v["speed"] >= 20, rng.uniform(-180, 180, n), np.nan)
    v["_time_since_wheel_onset"] = rng.uniform(0, 30, n)
    v["_time_since_wheel_offset"] = rng.uniform(0, 30, n)
    support = dict(camera=rng.random(n) < 0.98, wheel=np.ones(n, bool), all=np.ones(n, bool))
    return v, support


def test_features_are_well_formed():
    v, support = synthetic_behavior()
    discovery = np.arange(len(v["x"])) % 4 != 3
    built = features.build_features(v, support, discovery)
    assert len(built) == 23 and len({f.name for f in built}) == 23
    by_name = {f.name: f for f in built}
    for f in built:
        assert f.mask.dtype == bool and not (f.mask & ~f.valid).any()
        assert 0 < f.mask.sum() < f.valid.sum(), f.name
    headings = [by_name[f"heading_{d}"] for d in ("right", "down", "left", "up")]
    assert (np.sum([h.mask for h in headings], 0) == headings[0].valid).all()
    place = [by_name[k].mask for k in ("near_patch", "at_modal_location", "open_arena")]
    assert (np.sum(place, 0) <= 1 + (place[0] & place[1])).all()
    assert "(412.5, 237.5)" in by_name["at_modal_location"].definition
    # Thresholds from quantiles use the discovery part only.
    changed = {k: a.copy() for k, a in v.items()}
    changed["area"][~discovery] += 1e4
    again = {f.name: f for f in features.build_features(changed, support, discovery)}
    assert again["small_area"].definition == by_name["small_area"].definition
