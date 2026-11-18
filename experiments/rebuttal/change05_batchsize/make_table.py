"""Build the appendix table (LaTeX) and a plain-text summary from the two results files.

Run from the worktree root after analyze_churchland and analyze_aeon:
    python -m experiments.rebuttal.change05_batchsize.make_table
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TABLE = Path(__file__).resolve().parents[3] / 'paper/iclr_paper/figures/rebuttal/change05_batchsize/batchsize_table.tex'
BATCH_SIZES = (512, 1024, 2048)
FEATURE_NAMES = dict(recent_braking='Recent deceleration', fast_target_specific='Fast leftward target reach',
                     early_leftward='Early leftward movement', late_target_specific='Late, target-specific reach',
                     wheel='Wheel activity')
# A feature counts as lost at a batch size when, in at least two of three seeds, the best
# eligible latent's AUROC is more than this far below the paper model's (fixed in advance).
LOSS_MARGIN = .10


def fmt(values, digits=3, percent=False) -> str:
    """``mean [min--max]`` over seeds."""
    values = np.asarray(values, dtype=float) * (100 if percent else 1)
    digits = 0 if percent else digits
    return f'{values.mean():.{digits}f} [{values.min():.{digits}f}--{values.max():.{digits}f}]'


def single(value, digits=3, percent=False) -> str:
    value = float(value) * (100 if percent else 1)
    return f'{value:.{0 if percent else digits}f}'


# (criterion that picks the latent, metric reported, row label)
TWO_LATENTS = (('auroc', 'auroc', 'AUROC of best-AUROC latent'), ('sel', 'sel', 'Sel of best-Sel latent'))
ONE_LATENT = (('auroc', 'auroc', 'AUROC of best-AUROC latent'), ('auroc', 'sel', 'Sel of best-AUROC latent'))


def rows_for(models: dict, features: list, specs: tuple = TWO_LATENTS) -> list:
    """(label, paper value, {batch size: seed values}, kind) for every table row."""
    paper = next(m for m in models.values() if m['group'] == 'paper')
    runs = {b: [m for m in models.values() if m['group'] == 'retrained' and m['batch_size'] == b] for b in BATCH_SIZES}
    rows = []
    for name in features:
        rows.append(('header', FEATURE_NAMES[name], None, None))
        for criterion, metric, label in specs:
            rows.append(('value', label, paper['features'][name][criterion][metric],
                         {b: [m['features'][name][criterion][metric] for m in runs[b]] for b in BATCH_SIZES}))
    rows.append(('header', 'Dictionary', None, None))
    rows.append(('percent', 'Latents reappearing (\\%)', paper['match_share'],
                 {b: [m['match_share'] for m in runs[b]] for b in BATCH_SIZES}))
    rows.append(('count', 'Dead latents', paper['health']['dead_latents'],
                 {b: [m['health']['dead_latents'] for m in runs[b]] for b in BATCH_SIZES}))
    rows.append(('value', 'Variance explained', paper['health']['variance_explained'],
                 {b: [m['health']['variance_explained'] for m in runs[b]] for b in BATCH_SIZES}))
    return rows


def latex(datasets: list) -> str:
    lines = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrccc@{}}', r'\toprule',
             r'& Paper model & $B=512$ & $B=1{,}024$ & $B=2{,}048$ \\']
    for title, models, features, specs in datasets:
        lines += [r'\midrule', rf'\multicolumn{{5}}{{l}}{{\textbf{{{title}}}}} \\']
        for kind, label, paper, runs in rows_for(models, features, specs):
            if kind == 'header':
                lines.append(rf'\multicolumn{{5}}{{l}}{{\textit{{{label}}}}} \\')
                continue
            if kind == 'count':
                cells = [str(int(paper))] + [f'{np.mean(runs[b]):.1f} [{min(runs[b])}--{max(runs[b])}]' for b in BATCH_SIZES]
            else:
                percent = kind == 'percent'
                first = '--' if percent else single(paper)  # a model matches itself trivially
                cells = [first] + [fmt(runs[b], percent=percent) for b in BATCH_SIZES]
            lines.append(f'\\quad {label} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}']
    return '\n'.join(lines) + '\n'


def loss_flags(models: dict, features: list) -> list:
    """Feature/batch-size pairs where >=2 of 3 seeds lose more than LOSS_MARGIN best-AUROC."""
    paper = next(m for m in models.values() if m['group'] == 'paper')
    flags = []
    for name in features:
        reference = paper['features'][name]['auroc']['auroc']
        for b in BATCH_SIZES:
            runs = [m for m in models.values() if m['group'] == 'retrained' and m['batch_size'] == b]
            lost = [m['features'][name]['auroc']['auroc'] < reference - LOSS_MARGIN for m in runs]
            if sum(lost) >= 2:
                flags.append((name, b))
    return flags


def main() -> None:
    mac = json.loads((HERE / 'results/churchland_models.json').read_text())['models']
    aeon = json.loads((HERE / 'results/aeon_models.json').read_text())['models']
    mac_features = ['recent_braking', 'fast_target_specific', 'early_leftward', 'late_target_specific']
    datasets = [('Macaque reaching (Section 4.2; $D=192$, $k=16$)', mac, mac_features, TWO_LATENTS),
                ('Mouse foraging (Section 4.3; $D=192$, $k=12$)', aeon, ['wheel'], ONE_LATENT)]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(latex(datasets))
    flags = loss_flags(mac, mac_features) + loss_flags(aeon, ['wheel'])
    (HERE / 'results/loss_flags.json').write_text(json.dumps(flags))
    print('table written to', TABLE)
    print('features lost (>=2/3 seeds more than', LOSS_MARGIN, 'AUROC below paper):', flags or 'none')


if __name__ == '__main__':
    main()
