"""Measured scores must survive manuscript-display differences during reproduction."""
import json

from experiments.churchland import paper_results as report


def test_reproduced_table_uses_measured_scores_and_bolds_tied_maxima(tmp_path, monkeypatch):
    feature = 'recent_braking'
    selections, probes = [], []
    for i, method in enumerate(report.COLORS):
        for latent, criterion in enumerate(('illustrative',) if method == 'nldisco' else ('sel', 'auroc')):
            selections.append(dict(feature=feature, method=method, criterion=criterion,
                                   latent_id=latent, tpr=.4 + .1*i, fpr=.2,
                                   sel=(.4 + .1*i)/(.6 + .1*i), auroc=.7 + .05*i))
            probes.append(dict(feature=feature, method=method, latent_id=latent,
                               test={'balanced_accuracy': .7}, interval={}))
        probes.append(dict(feature=feature, method=method, latent_id=None,
                           test={'balanced_accuracy': .8}, interval={}))
    (tmp_path/'selection.json').write_text(json.dumps({'selections': selections}))
    (tmp_path/'results.json').write_text(json.dumps({'results': probes}))
    monkeypatch.setattr(report, 'SOURCE', tmp_path)
    monkeypatch.setattr(report, 'FEATURE_ORDER', (feature,))
    monkeypatch.setattr(report, 'main_table_values', lambda _: {
        (feature, 'nldisco', 'illustrative'): ['0.9900'] * 6})
    records = report.render_table(figures=tmp_path, decode_output=tmp_path)
    assert records[0]['displayed_scores'][0] == '0.4000'
    assert records[0]['manuscript_scores'][0] == '0.9900'
    table = (tmp_path/'churchland_feature_metrics.tex').read_text()
    assert '0.9900' not in table
    assert table.count(r'\textbf{0.6000}') == 2  # Both tied selectors remain bold.
    assert table.count(r'\textbf{0.8000}') == 10  # Tied specificity and all-space Decode.
