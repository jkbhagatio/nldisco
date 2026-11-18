"""Fast checks for the threshold-sensitivity analysis."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from experiments.churchland.sweeps.s10_endpoint_v1.comparison_features import FeatureData, score_dense
from experiments.rebuttal.change09_thresholds import threshold_sensitivity as ts


def toy(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    trials = rng.integers(0, 60, n)
    y = rng.random(n) < .2
    x = rng.normal(size=n) + 1.5 * y
    return x, y, trials


def test_rate_threshold_matches_requested_rate_and_paper_rule():
    x, y, trials = toy()
    feature = FeatureData(np.arange(len(x)), y, trials, .13, {})
    row = score_dense(x[:, None], feature)[0]
    out = ts.score_at_rate(x * row['orientation'], y, .13)
    assert out['achieved_rate'] == pytest.approx(.13, abs=1e-3)
    assert out['cov'] == pytest.approx(row['tpr']) and 1 - out['spec'] == pytest.approx(row['fpr'])
    assert out['sel'] == pytest.approx(row['sel'])


def test_curve_monotone_in_cov_and_spec():
    x, y, _ = toy()
    curve = ts.rate_curve(x, y, np.geomspace(.01, .9, 20))
    cov, spec = [c['cov'] for c in curve], [c['spec'] for c in curve]
    assert np.all(np.diff(cov) >= 0) and np.all(np.diff(spec) <= 0)


def brute_best(x, y, trials, min_cov):
    best = None
    for t in np.unique(x):
        a = x > t
        tpr, fpr = a[y].mean(), a[~y].mean()
        support = len(np.unique(trials[a & y]))
        if tpr >= min_cov and support >= ts.MIN_TRIALS:
            key = (ts.selectivity(tpr, fpr), tpr)
            best = key if best is None or key > best else best
    return best


@pytest.mark.parametrize('min_cov', [.05, .3])
def test_best_threshold_is_exact(min_cov):
    x, y, trials = toy(1)
    out = ts.best_threshold(x, y, trials, min_cov)
    sel, cov = brute_best(x, y, trials, min_cov)
    assert out['eligible'] and out['sel'] == pytest.approx(sel) and out['cov'] == pytest.approx(cov)


def test_best_threshold_ineligible_and_ties():
    x, y, trials = toy()
    assert not ts.best_threshold(x, y, trials, 1.01)['eligible']
    sparse = np.where(x > 1.0, x, 0.0)  # many exact zeros, like a BatchTopK latent
    out = ts.best_threshold(sparse, y, trials)
    assert out['eligible'] and out['threshold'] >= 0


def test_auroc_ignores_threshold_and_ties():
    x, y, _ = toy()
    x = np.round(x, 1)
    assert ts.auroc(x, y) == pytest.approx(roc_auc_score(y, x))


def test_sparse_latent_has_no_double_setting():
    x, y, trials = toy()
    x = np.where(x > 1.0, x, 0.0)
    feature = FeatureData(np.arange(len(x)), y, trials, .2, {})
    native = float((x > 0).mean())
    out = ts.settings_for(x, feature, native, dense=False)
    assert out['double'] is None and out['paper']['achieved_rate'] == pytest.approx(native)
    assert out['half']['achieved_rate'] == pytest.approx(native / 2, abs=2e-3)


def test_selected_latents_merges_identical_picks_and_filters_nldisco():
    rows = [dict(method='cebra', feature='early_leftward', criterion=c, latent_id=3, orientation=1)
            for c in ('sel', 'auroc')]
    rows += [dict(method='nldisco', feature='recent_braking', criterion=c, latent_id=i, orientation=1)
             for c, i in (('sel', 1), ('auroc', 2), ('illustrative', 83))]
    spec = dict(selections=rows, main_features=['recent_braking'])
    cebra = ts.selected_latents(spec, ts.METHODS[1], np.zeros((1, 1)), {})
    assert len(cebra) == 1 and cebra[0]['selectors'] == ['sel', 'auroc']
    nldisco = ts.selected_latents(spec, ts.METHODS[0], np.zeros((1, 1)), {})
    assert [r['latent_id'] for r in nldisco] == [83]


def test_extra_method_parses_to_paper_rule_entry():
    config = ts.parse_extra('stndt:ST-NDT:/x/y.npz:#9467bd')
    assert config.key == 'stndt' and config.selections == 'paper_rule'
    assert config.representation == {'path': '/x/y.npz'} and config.dense
