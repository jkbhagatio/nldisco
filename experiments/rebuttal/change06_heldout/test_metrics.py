"""Fast tests of the metric, selection and split helpers."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from experiments.rebuttal.change06_heldout import aeon_split, metrics


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    values = np.maximum(rng.normal(size=(500, 6)), 0).astype(np.float32)
    labels = rng.random(500) < .3
    ours = metrics.auroc_columns(values, labels)
    ref = [roc_auc_score(labels, values[:, i]) for i in range(6)]
    np.testing.assert_allclose(ours, ref)


def test_rate_metrics_definitions():
    active = np.array([[1, 1, 0, 0, 1, 0]], dtype=bool).T
    labels = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    out = metrics.rate_metrics(active, labels)
    assert out["cov"][0] == pytest.approx(2 / 3)
    assert out["spec"][0] == pytest.approx(2 / 3)
    assert out["sel"][0] == pytest.approx((2 / 3) / (2 / 3 + 1 / 3))


def test_selection_uses_only_given_rows_and_eligibility():
    rng = np.random.default_rng(1)
    n = 4000
    labels = rng.random(n) < .2
    groups = np.arange(n) // 10
    values = np.zeros((n, 3), dtype=np.float32)
    values[:, 0] = (rng.random(n) < .5) * rng.random(n)           # uninformative
    values[:, 1] = labels * rng.random(n) + (rng.random(n) < .05)  # informative
    values[:, 2] = (labels & (rng.random(n) < .01)) * 1.0          # perfect but ineligible
    table = metrics.score_latents(values, labels, groups)
    assert not table["eligible"][2]
    assert metrics.select_latent(table, "auroc") == 1
    assert metrics.select_latent(table, "sel") == 1


def test_selection_independent_of_heldout_rows():
    rng = np.random.default_rng(2)
    n = 3000
    labels = rng.random(n) < .25
    groups = np.arange(n) // 7
    values = np.maximum(rng.normal(size=(n, 5)) + labels[:, None] * np.array([0, 1, 0, 2, 0]), 0).astype(np.float32)
    a = metrics.select_latent(metrics.score_latents(values[:2000], labels[:2000], groups[:2000]), "auroc")
    corrupted = values.copy()
    corrupted[2000:] = rng.random((1000, 5))
    b = metrics.select_latent(metrics.score_latents(corrupted[:2000], labels[:2000], groups[:2000]), "auroc")
    assert a == b


def test_operating_threshold_hits_target_fpr():
    rng = np.random.default_rng(3)
    labels = rng.random(20000) < .1
    values = rng.normal(size=20000) + 1.5 * labels
    t = metrics.operating_threshold(values, labels, .18)
    assert abs((values[~labels] >= t).mean() - .18) < 2e-3


def test_split_partitions_and_boundaries():
    held = aeon_split.heldout_bins()
    assert held.mean() == pytest.approx(14 / 72)
    starts = np.arange(0, 3_240_000 - 20, 7)
    label = aeon_split.partition_windows(starts, held)
    assert set(np.unique(label)) == {-1, 0, 1}
    for s, lab in zip(starts[::997], label[::997]):
        bins = held[s:s + 20]
        assert lab == (1 if bins.all() else 0 if not bins.any() else -1)
    # train windows never overlap any held-out bin
    train = starts[label == 0]
    assert not any(held[s:s + 20].any() for s in train[::101])


def test_episode_ids_and_summary():
    ids = metrics.episode_ids(np.ones(6, bool), np.array([0, 1, 2, 80, 81, 200]))
    assert ids.tolist() == [0, 0, 0, 1, 1, 2]
    s = metrics.summarize([1., 2., 3.])
    assert s["mean"] == 2 and s["sd"] == pytest.approx(1.0)


def test_evaluate_feature_selects_on_train_and_scores_heldout():
    from experiments.rebuttal.change06_heldout import scoring
    rng = np.random.default_rng(4)
    n = 6000
    part = (np.arange(n) >= 4500).astype(np.int8)
    labels = rng.random(n) < .2
    groups = np.arange(n) // 10
    values = np.maximum(rng.normal(size=(n, 6)) - 1, 0).astype(np.float32)
    values[:, 3] += labels * rng.random(n).astype(np.float32) * 2  # the true feature latent
    out = scoring.evaluate_feature(values, labels, part, groups, operating_fpr=.18)
    assert out["auroc"]["latent"] == 3
    rec = out["auroc"]
    assert rec["heldout"]["auroc"] > .7 and abs(rec["heldout"]["auroc"] - rec["train"]["auroc"]) < .1
    assert abs(rec["train_operating"]["fpr"] - .18) < .01
    # changing held-out values cannot change the selected latent or training scores
    changed = values.copy()
    changed[part == 1] = rng.random((int((part == 1).sum()), 6)).astype(np.float32)
    out2 = scoring.evaluate_feature(changed, labels, part, groups)
    assert out2["auroc"]["latent"] == out["auroc"]["latent"]
    assert out2["auroc"]["train"] == out["auroc"]["train"]
