"""Fast tests for the batch-size study helpers (CPU only)."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from experiments.rebuttal.change05_batchsize import scoring
from experiments.rebuttal.change05_batchsize.train_churchland import training_config


def sparse_data(seed=0, n=600, d=7):
    rng = np.random.default_rng(seed)
    x = rng.exponential(size=(n, d)).astype(np.float32) * (rng.random((n, d)) < .3)
    condition = rng.random(n) < .25
    x[:, 0] += 2. * condition * (rng.random(n) < .7)  # latent 0 detects the condition
    x[:, 6] = 0.  # a dead latent
    return x.astype(np.float32), condition


def test_auroc_matches_sklearn_with_ties():
    x, condition = sparse_data()
    ours = scoring.auroc_columns(x, condition, device='cpu')
    expected = [roc_auc_score(condition, x[:, j]) if x[:, j].ptp() else .5 for j in range(x.shape[1])]
    np.testing.assert_allclose(ours, expected, atol=1e-9)


def test_score_latents_and_eligibility():
    x, condition = sparse_data()
    trials = np.repeat(np.arange(30), 20)
    scores = scoring.score_latents(x, condition, trials, device='cpu')
    active = x > 0
    np.testing.assert_allclose(scores['tpr'], active[condition].mean(0))
    np.testing.assert_allclose(scores['sel'][0], scores['tpr'][0] / (scores['tpr'][0] + scores['fpr'][0]))
    assert not scores['eligible'][6] and not scores['varying'][6]
    assert scores['eligible'][0]
    assert scoring.best_latent(scores, 'auroc') == 0


def test_best_latent_ties_and_none():
    scores = dict(sel=np.array([.5, .9, .9, .95]), auroc=np.array([.6, .7, .7, .5]),
                  eligible=np.array([True, True, True, False]))
    assert scoring.best_latent(scores, 'sel') == 1
    assert scoring.best_latent(scores, 'auroc') == 1
    scores['eligible'][:] = False
    assert scoring.best_latent(scores, 'sel') is None
    with pytest.raises(ValueError):
        scoring.best_latent(scores, 'other')


def test_trial_support_counts_distinct_trials():
    x = np.zeros((40, 1), dtype=np.float32)
    x[:10, 0] = 1.
    condition = np.zeros(40, dtype=bool)
    condition[:20] = True
    trials = np.repeat(np.arange(8), 5)
    scores = scoring.score_latents(x, condition, trials, device='cpu')
    assert scores['support'][0] == 2  # active in trials 0 and 1 only


def test_column_correlation_matches_numpy_and_matches():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(1000, 5)).astype(np.float32)
    b = np.concatenate([a[:, [2, 0]] + .1 * rng.normal(size=(1000, 2)),
                        rng.normal(size=(1000, 3)), np.zeros((1000, 1))], axis=1).astype(np.float32)
    corr = scoring.column_correlation(a, b, chunk=300, device='cpu')
    expected = np.corrcoef(a.T, b[:, :5].T)[:5, 5:]
    np.testing.assert_allclose(corr[:, :5], expected, atol=1e-6)
    assert (corr[:, 5] == 0).all()
    matched = scoring.match_latents(corr, threshold=.9)
    assert matched['index'][2] == 0 and matched['index'][0] == 1
    assert matched['share'] == pytest.approx(2 / 5)


def test_dead_and_variance_explained():
    x, _ = sparse_data()
    assert scoring.dead_latents(x) == 1
    assert scoring.variance_explained(np.array([.5, .5]), np.array([1., 1.])) == pytest.approx(.5)


def test_training_config_fixes_examples_and_scales_with_epoch_fraction():
    for batch_size in (512, 1024, 2048):
        n_batches = -(-130507 // batch_size)
        config = training_config(batch_size, n_batches)
        assert config.epochs * n_batches * batch_size >= 10 * 130507
        assert config.dead_feature_window == n_batches // 3
    assert training_config(1024, 128).log_frequency == 32 and training_config(1024, 128).dead_feature_window == 42
