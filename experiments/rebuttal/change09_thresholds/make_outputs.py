"""Figure and table for the threshold-sensitivity appendix from ``threshold_sensitivity.json``.

Run from the worktree root::

    $PY -m experiments.rebuttal.change09_thresholds.make_outputs --results <json> --out <dir>
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

from experiments.plot_style import format_plot_label
from experiments.rebuttal.change09_thresholds.threshold_sensitivity import FEATURE_ORDER

TITLES = dict(recent_braking='Recent deceleration', fast_target_specific='Fast leftward target reach',
              early_leftward='Early leftward movement', late_target_specific='Late, target-specific reach')
METRICS = (('cov', 'Cov'), ('spec', 'Spec'), ('sel', 'Sel'))
SELECTOR = dict(sel='best Sel', auroc='best AUROC', illustrative='illustrative')


def selector_text(selectors: list) -> str:
    """Selector names of one latent, e.g. ``best Sel/AUROC``."""
    names = [SELECTOR[s] for s in selectors]
    return names[0] if len(names) == 1 else 'best Sel/AUROC' if 'sel' in selectors else '/'.join(names)


def draw(report: dict, out: Path) -> None:
    """Rows: Cov, Spec, Sel; columns: the four features; x: activation rate (log)."""
    colors = {m['key']: m['color'] for m in report['methods']}
    fig, axes = plt.subplots(3, 4, figsize=(7.4, 5.6), sharex='col')
    for col, feature in enumerate(FEATURE_ORDER):
        rate = report['paper_rates'][feature]
        group = [r for r in report['results'] if r['feature'] == feature]
        for row, (metric, label) in enumerate(METRICS):
            ax = axes[row, col]
            ax.axvline(rate, color='0.5', lw=.8, ls=':')
            seen = {}
            for r in group:
                style = '-' if seen.setdefault(r['method'], 0) == 0 else '--'
                seen[r['method']] += 1
                curve = r['curve']
                ax.plot([c['achieved_rate'] for c in curve], [c[metric] for c in curve],
                        color=colors[r['method']], ls=style, lw=1.3)
                p = r['settings']['paper']
                ax.plot(p['achieved_rate'], p[metric], 'o', color=colors[r['method']], ms=3.5,
                        mec='white', mew=.5, zorder=4)
            ax.set_xscale('log')
            ax.set_ylim(-.03, 1.03)
            ax.grid(alpha=.25, lw=.4)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(rf'$\mathbf{{{label}}}$', fontsize=8)
            if row == 0:
                ax.set_title(TITLES[feature], fontsize=7.5, style='italic')
            if row == 2:
                ax.set_xlabel('Activation rate', fontsize=7.5)
    handles = [Line2D([], [], color=m['color'], lw=1.5, label=m['label']) for m in report['methods']]
    handles += [Line2D([], [], color='k', ls='-', lw=1.2, label='best-Sel / illustrative latent'),
                Line2D([], [], color='k', ls='--', lw=1.2, label='best-AUROC latent'),
                Line2D([], [], color='k', marker='o', ls='', ms=3.5, label="paper's operating point"),
                Line2D([], [], color='0.5', ls=':', lw=.9, label="paper's rate $r$ (dense baselines)")]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(.5, -.005))
    fig.tight_layout(rect=(0, .07, 1, 1), h_pad=.6, w_pad=.5)
    titles = {ax.title for ax in axes.ravel()}
    for text in fig.findobj(match=Text):
        if text not in titles:  # titles keep plain italics (the style helper would typeset '-' as minus)
            text.set_text(format_plot_label(text.get_text()))
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'threshold_sensitivity.{ext}', dpi=210)
    plt.close(fig)


def cells(d) -> str:
    """Cov / Spec / Sel cells of one setting (``--`` when unreachable)."""
    if d is None or not d.get('eligible', True):
        return ' & '.join(['--'] * 3)
    return ' & '.join(f"{d[m]:.3f}" for m in ('cov', 'spec', 'sel'))


def table(report: dict) -> str:
    """LaTeX tabular with Cov/Spec/Sel at half, paper, twice the rate, and own best threshold."""
    heads = [('half', r'$r/2$'), ('paper', r'Paper $r$'), ('double', r'$2r$'), ('best', 'Own best')]
    lines = [r'\begin{tabular}{llr' + 'rrr' * 4 + r'r}', r'\toprule',
             r'Latent & Selector & \metric{AUROC} & ' + ' & '.join(
                 rf'\multicolumn{{3}}{{c}}{{{h}}}' for _, h in heads) + r' & Rate \\',
             ''.join(rf'\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}' for i in range(4)),
             ' & & & ' + ' & '.join([r'\metric{Cov} & \metric{Spec} & \metric{Sel}'] * 4)
             + r' & at best \\', r'\midrule']
    for k, feature in enumerate(FEATURE_ORDER):
        if k:
            lines.append(r'\midrule')
        lines.append(rf'\multicolumn{{15}}{{l}}{{\textit{{{TITLES[feature]}}}; $r={report["paper_rates"][feature]*100:.2f}\%$}} \\')
        for r in (x for x in report['results'] if x['feature'] == feature):
            s = r['settings']
            best = s['best']
            rate = f"{best['achieved_rate'] * 100:.2f}\\%" if best.get('eligible', True) else '--'
            lines.append(f"{r['label']} {r['latent_id']} & {selector_text(r['selectors'])} & "
                         f"{r['auroc']:.3f} & " + ' & '.join(cells(s[n]) for n, _ in heads)
                         + f' & {rate} \\\\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}']) + '\n'


def matched_table(report: dict, reference: dict) -> str:
    """Equal-coverage comparison: each baseline at its best-Sel threshold with Cov >= the
    reference NLDisco latent's coverage; NLDisco at its native threshold."""
    lines = [r'\begin{tabular}{llrrr}', r'\toprule',
             r'Latent & Selector & \metric{Cov} & \metric{Spec} & \metric{Sel} \\',
             r'\midrule']
    for k, feature in enumerate(FEATURE_ORDER):
        if k:
            lines.append(r'\midrule')
        lines.append(rf'\multicolumn{{5}}{{l}}{{\textit{{{TITLES[feature]}}}}} \\')
        for r in (x for x in report['results'] if x['feature'] == feature):
            s = r['settings']
            if not r['dense']:
                if r['latent_id'] != reference[feature]:
                    continue
                d = s['paper']
            else:
                d = s['matched']
            lines.append(f"{r['label']} {r['latent_id']} & {selector_text(r['selectors'])} & "
                         f"{d['cov']:.3f} & {d['spec']:.3f} & {d['sel']:.3f} \\\\")
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}']) + '\n'


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.results.read_text())
    draw(report, args.out)
    (args.out / 'threshold_sensitivity_table.tex').write_text(table(report))
    (args.out / 'threshold_sensitivity_matched_table.tex').write_text(
        matched_table(report, report['reference_latents']))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
