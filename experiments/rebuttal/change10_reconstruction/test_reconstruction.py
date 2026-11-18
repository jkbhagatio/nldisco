"""Fast unit tests for the change-10 reconstruction scores and shared Poisson readout."""
import json

import numpy as np
import pytest
import torch
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from experiments.rebuttal.change10_reconstruction import reconstruction as rc


@pytest.fixture
def glm_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 6))
    w = rng.normal(scale=.4, size=(6, 4))
    counts = rng.poisson(np.exp(x @ w - 1)).astype(np.float64)
    return x, counts, np.exp(x @ w - 1)


def test_mean_predictor_scores_zero(glm_data):
    _, counts, _ = glm_data
    mean = np.broadcast_to(counts.mean(0), counts.shape).copy()
    assert rc.bits_per_spike(mean, counts) == pytest.approx(0, abs=1e-12)
    assert rc.variance_explained(mean, counts)[0] == pytest.approx(0, abs=1e-12)


def test_bits_per_spike_matches_definition(glm_data):
    _, counts, rates = glm_data
    ll = poisson.logpmf(counts, rates).sum() - poisson.logpmf(counts, counts.mean(0)).sum()
    assert rc.bits_per_spike(rates, counts) == pytest.approx(ll / counts.sum() / np.log(2))
    assert rc.bits_per_spike(rates, counts) > 0


def test_variance_explained_skips_silent_channels():
    counts = np.array([[0., 1.], [0., 3.], [0., 2.]])
    ve, skipped = rc.variance_explained(counts.copy(), counts)
    assert (ve, skipped) == (1.0, 1)


def test_zero_rate_for_observed_spike_raises():
    with pytest.raises(ValueError):
        rc.bits_per_spike(np.array([[0.], [1.]]), np.array([[1.], [1.]]))


def test_fit_poisson_matches_sklearn(glm_data):
    x, counts, _ = glm_data
    alpha = 1e-2
    w, b, grad = rc.fit_poisson(torch.as_tensor(x), torch.as_tensor(counts), alpha)
    assert grad < 1e-6
    for c in range(counts.shape[1]):
        ref = PoissonRegressor(alpha=alpha, max_iter=1000, tol=1e-10).fit(x, counts[:, c])
        np.testing.assert_allclose(w[:, c].numpy(), ref.coef_, atol=1e-5)
        assert b[c].item() == pytest.approx(ref.intercept_, abs=1e-5)


def test_shared_readout_selects_on_validation(glm_data):
    x, counts, rates = glm_data
    split = np.repeat([0, 1, 2], [2000, 500, 500])
    pred, record = rc.shared_readout(x, counts, split, 'cpu', alphas=(1e-4, 1e-1, 10.))
    best = max(record['grid'], key=lambda r: r['validation_bits_per_spike'])
    assert record['chosen_alpha'] == best['alpha'] and pred.shape == (500, 4)
    assert rc.bits_per_spike(pred, counts[2500:]) > .9 * rc.bits_per_spike(rates[2500:], counts[2500:])


def test_native_readout_floor_only_when_needed(glm_data):
    _, counts, rates = glm_data
    split = np.repeat([0, 1, 2], [2000, 500, 500])
    out, info = rc.native_readout(rates, counts, split)
    assert info['rate_floor'] == 0 and np.array_equal(out, rates[2500:])
    shifted = rates - .05
    out, info = rc.native_readout(shifted, counts, split)
    assert info['rate_floor'] in rc.FLOORS and out.min() >= info['rate_floor']


def test_load_values_checks_index(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, 'N_WINDOWS', 5)
    anchors = np.arange(9, 14)
    np.save(tmp_path / 'acts.npy', np.ones((5, 3), np.float32))
    np.savez(tmp_path / 'index.npz', anchor_index=anchors)
    assert rc.load_values(tmp_path / 'acts.npy', anchors).shape == (5, 3)
    with pytest.raises(AssertionError):
        rc.load_values(tmp_path / 'acts.npy', anchors + 1)


def test_stndt_marker(tmp_path):
    assert rc.stndt_method(tmp_path / 'missing.json') is None
    marker = tmp_path / 'REPRESENTATIONS_READY.json'
    marker.write_text(json.dumps(dict(path='/x/acts.npy', shape=[130507, 192])))
    method = rc.stndt_method(marker)
    assert (method.key, method.representation, method.native) == ('stndt', '/x/acts.npy', None)


def test_latex_table_bolds_best(tmp_path):
    def entry(label, v, native=None):
        test = dict(bits_per_spike=v, variance_explained=v, bits_per_spike_ci=[0, 1], variance_explained_ci=[0, 1])
        return dict(label=label, readout=dict(test=test),
                    native=None if native is None else dict(test={**test, 'bits_per_spike': native}))
    rc.latex_table(dict(a=entry('A', .1, .3), b=entry('B', .2)), tmp_path / 't.tex')
    a, b = (tmp_path / 't.tex').read_text().splitlines()
    assert r'\textbf{0.200}' in b and '--' in b and r'\textbf{0.300}' in a


def test_trial_bootstrap_matches_naive_resampling(glm_data):
    _, counts, rates = glm_data
    trials = np.repeat(np.arange(60) * 7, 50)
    out = rc.trial_bootstrap(rates, counts, trials, repetitions=40, seed=3)
    blocks = [np.flatnonzero(trials == t) for t in np.unique(trials)]
    rng, draws = np.random.default_rng(3), []
    for _ in range(40):
        rows = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        draws.append((rc.bits_per_spike(rates[rows], counts[rows]), rc.variance_explained(rates[rows], counts[rows])[0]))
    low, high = np.quantile(np.asarray(draws), [.025, .975], axis=0)
    np.testing.assert_allclose(out['bits_per_spike_ci'], [low[0], high[0]], rtol=1e-9)
    np.testing.assert_allclose(out['variance_explained_ci'], [low[1], high[1]], rtol=1e-9)


def test_latex_table_bolds_ties_at_displayed_precision(tmp_path):
    test = lambda v: dict(bits_per_spike=v, variance_explained=v, bits_per_spike_ci=[0, 1], variance_explained_ci=[0, 1])  # noqa: E731
    rc.latex_table(dict(a=dict(label='A', readout=dict(test=test(.33511)), native=None),
                        b=dict(label='B', readout=dict(test=test(.33497)), native=None)), tmp_path / 't.tex')
    assert all(r'\textbf{0.335}' in line for line in (tmp_path / 't.tex').read_text().splitlines())
