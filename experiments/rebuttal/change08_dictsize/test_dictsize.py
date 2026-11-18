"""Fast tests for the dictionary-size analysis helpers."""
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from experiments.rebuttal.change08_dictsize import dictsize as ds


def _acts(n=400, d=12, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.exponential(size=(n, d)) * (rng.random((n, d)) < .15)
    x[:, 3] = 0.  # dead latent
    return x.astype(np.float32)


def test_pearson_cross_matches_numpy():
    a, b = _acts(seed=0), _acts(d=7, seed=1)
    got = ds.pearson_cross(sp.csr_matrix(a), sp.csr_matrix(b))
    want = np.corrcoef(a.T, b.T)[:12, 12:]
    live = (a.std(0) > 0)[:, None] & (b.std(0) > 0)[None]
    np.testing.assert_allclose(got[live], want[live], atol=1e-5)
    assert (got[~live] == 0).all()


def test_score_latents_matches_sklearn():
    x = _acts()
    rng = np.random.default_rng(2)
    rows = np.sort(rng.choice(len(x), 300, replace=False))
    cond = rng.random(300) < .3
    cond |= x[rows, 5] > 0  # make latent 5 informative
    groups = rng.integers(0, 40, 300)
    got = ds.score_latents(sp.csr_matrix(x), rows, cond, groups)
    for j in (0, 5, 9):
        v = x[rows, j]
        assert np.isclose(got.auroc[j], roc_auc_score(cond, v))
        tpr, fpr = (v[cond] > 0).mean(), (v[~cond] > 0).mean()
        assert np.isclose(got.sel[j], tpr / (tpr + fpr))
        assert got.support[j] == len(np.unique(groups[(v > 0) & cond]))
    assert not got.eligible[3] and got.varying[3] == 0  # dead latent never eligible
    assert got.auroc[5] > got.auroc[0]


def test_select_winners_ties_use_lowest_id():
    frame = ds.pd.DataFrame(dict(latent=[0, 1, 2], sel=[.5, .9, .9], auroc=[.6, .6, .7],
                                 eligible=[True, True, False]))
    winners = ds.select_winners(frame)
    assert winners['sel']['latent'] == 1 and winners['auroc']['latent'] == 0


def test_reappearance_and_new_counts():
    corr = np.array([[.9, 0, 0], [0, .2, .3], [0, 0, 0]])
    out = ds.reappearance(np.array([True, True, False]), corr, np.array([True, True, True]), .5)
    assert out['n_shared'] == 1 and out['n_ref_live'] == 2
    assert out['n_new'] == 2  # big latents 1 and 2 have no reference partner >= .5


def test_near_duplicates_counts_clusters():
    c = np.eye(5)
    c[0, 1] = c[1, 0] = c[1, 2] = c[2, 1] = .9  # chain 0-1-2
    out = ds.near_duplicates(c, np.ones(5, bool), .8)
    assert out['n_near_duplicate'] == 3 and out['n_redundant_removable'] == 2


def test_run_groups_splits_gaps_and_polarity():
    rows = np.array([0, 1, 2, 5, 6, 7])
    pos = np.array([True, True, False, True, True, True])
    g = ds.run_groups(pos, rows)
    assert g[0] == g[1] and g[3] == g[4] == g[5] and g[0] != g[3] and g[2] == -1


def test_report_tidy_and_summary(tmp_path):
    import json
    from experiments.rebuttal.change08_dictsize import report

    def record(d, seed, paper):
        feat = {n: dict(n_eligible=3, auroc=dict(latent=1., auroc=.9, sel=.7), sel=dict(latent=2., auroc=.6, sel=.99))
                for n in report.FEATURES['aeon']}
        r = dict(dataset='aeon', D=d, seed=seed, role='paper' if paper else 'new', dead=d // 10,
                 dead_fraction=.1, mean_l0=12., train_seconds=10. * d, is_paper_model=paper, features=feat,
                 near_duplicates={t: dict(n_near_duplicate=5, share_near_duplicate=.05, n_redundant_removable=3, n_live=d)
                                  for t in ('0.5', '0.7', '0.8', '0.9')})
        if not paper:
            r['vs_paper_model'] = {t: dict(share_reappearing=.9, n_shared=170, n_new=40, share_new=.2)
                                   for t in ('0.3', '0.5', '0.7', '0.9')}
        (tmp_path / f'aeon_d{d}_s{seed}.json').write_text(json.dumps(r))

    record(192, 0, True)
    record(512, 0, False)
    record(512, 1, False)
    frame = report.tidy(tmp_path)
    table = report.summary_table(frame, 'aeon')
    assert list(table.D) == [192, 512] and table.n_runs.tolist() == [1, 2]
    assert table.loc[table.D == 512, 'shared'].item() == .9 and np.isnan(table.loc[table.D == 192, 'shared'].item())
    assert table.loc[table.D == 512, 'dead_mean'].item() == 51
