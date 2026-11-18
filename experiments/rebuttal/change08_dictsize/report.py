"""Aggregate per-run analyses into tables and the appendix figure.

    python -m experiments.rebuttal.change08_dictsize.report \
        --analysis <ceph>/analysis --figure-dir paper/iclr_paper/figures/rebuttal/change08_dictsize
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CEPH = Path('/ceph/aeon/aeon/nldisco/rebuttal_20261118/change08_dictsize')
FEATURES = dict(
    macaque=('recent_braking', 'early_leftward', 'fast_target_specific', 'late_target_specific'),
    aeon=('wheel', 'area', 'speed', 'direction'), session_b=())
LABELS = dict(
    macaque=('Recent deceleration', 'Early leftward', 'Fast leftward (target)', 'Late reach (target)'),
    aeon=('Wheel activity', 'Smaller area near patch', 'Fast centroid speed', 'Fast outward movement'))
TITLES = dict(macaque='Macaque (Sec. 4.2)', aeon='Aeon (Sec. 4.3)', session_b='Aeon session B')
TAU, TAU_DUP = '0.5', '0.8'


def tidy(analysis: Path) -> pd.DataFrame:
    """One row per (dataset, D, seed) from the per-run JSON records."""
    rows = []
    for path in sorted(analysis.glob('*.json')):
        r = json.loads(path.read_text())
        row = dict(dataset=r['dataset'], D=r['D'], seed=r['seed'], role=r['role'], dead=r['dead'],
                   dead_frac=r['dead_fraction'], live=r['D'] - r['dead'], mean_l0=r['mean_l0'],
                   train_seconds=r['train_seconds'], paper_model=r['is_paper_model'])
        for name, f in r['features'].items():
            for rule in ('auroc', 'sel'):
                w = f[rule]
                row[f'{name}_{rule}_latent'] = None if w is None else int(w['latent'])
                row[f'{name}_{rule}_sel'] = np.nan if w is None else w['sel']
                row[f'{name}_{rule}_auroc'] = np.nan if w is None else w['auroc']
        dup = r['near_duplicates'][TAU_DUP]
        row.update(dup=dup['n_near_duplicate'], dup_share=dup['share_near_duplicate'])
        for tau, v in r.get('vs_paper_model', {}).items():
            row.update({f'shared_{tau}': v['share_reappearing'], f'n_shared_{tau}': v['n_shared'],
                        f'new_{tau}': v['n_new'], f'new_share_{tau}': v['share_new']})
        for tau, v in r['near_duplicates'].items():
            row[f'dup_{tau}'] = v['n_near_duplicate']
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['dataset', 'D', 'seed']).reset_index(drop=True)


def timing(path: Path, dataset: str) -> pd.DataFrame:
    """Schedule-time estimates per D from the interleaved micro-benchmark (min over repeats)."""
    from experiments.rebuttal.change08_dictsize.time_training import SCHEDULE, estimate
    if not path.exists():
        return pd.DataFrame(columns=['D', 'seconds', 'seconds_median'])
    records = json.loads(path.read_text())
    if not any(not r['warmup'] for r in records):
        return pd.DataFrame(columns=['D', 'seconds', 'seconds_median'])
    est = estimate(records, dataset)
    plan = SCHEDULE[dataset]
    return pd.DataFrame([dict(D=d, seconds=v['schedule_seconds'], update_ms=v['update_seconds'] * 1e3,
                              seconds_median=plan['updates'] * v['median_update_seconds']
                              + plan['calibration'] * v['calibration_batch_seconds'])
                         for d, v in est.items()])


def summary_table(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Mean over all runs of each size (macaque D=192: the paper run and five further seeds)."""
    sub = frame[frame.dataset == dataset]
    out = []
    for d, g in sub.groupby('D'):
        row = dict(D=d, n_runs=len(g), dead_mean=g.dead.mean(), dead_min=g.dead.min(), dead_max=g.dead.max(),
                   time_s_mean=g.train_seconds.mean())
        scored = g
        for name in FEATURES[dataset]:
            row[f'{name}_auroc'] = scored[f'{name}_auroc_auroc'].mean()
            row[f'{name}_sel'] = scored[f'{name}_auroc_sel'].mean()
        others = g[~g.paper_model]
        row['shared'] = others[f'shared_{TAU}'].mean() if len(others) else np.nan
        row['new_share'] = others[f'new_share_{TAU}'].mean() if len(others) else np.nan
        row['new'] = others[f'new_{TAU}'].mean() if len(others) else np.nan
        row['dup'] = g.dup.mean()
        row['dup_share'] = g.dup_share.mean()
        out.append(row)
    return pd.DataFrame(out)


def markdown(frame: pd.DataFrame, dataset: str, times: pd.DataFrame) -> str:
    """Per-size markdown table: mean (min-max) over runs where more than one run exists."""
    def span(g, col, fmt='{:.0f}'):
        v = g[col].dropna()
        if v.empty:
            return '-'
        return fmt.format(v.mean()) + ('' if len(v) == 1 else ' (' + fmt.format(v.min()) + '-' + fmt.format(v.max()) + ')')
    sub = frame[frame.dataset == dataset]
    head = ['D', 'runs', 'dead', 'dead %', 'time [s] (recorded)', 'time [s] (benchmark)']
    head += [f'{n} AUROC / Sel' for n in FEATURES[dataset]]
    head += ['paper latents reappearing %', 'new live latents', 'new %', 'near-dup. live latents', 'near-dup. %']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for d, g in sub.groupby('D'):
        bench = times[times.D == d].seconds
        row = [str(d), str(len(g)), span(g, 'dead'), span(g.assign(p=g.dead_frac * 100), 'p', '{:.1f}'),
               span(g, 'train_seconds'), '-' if bench.empty else f'{bench.iloc[0]:.0f}']
        for n in FEATURES[dataset]:
            row.append(span(g, f'{n}_auroc_auroc', '{:.3f}') + ' / ' + span(g, f'{n}_auroc_sel', '{:.3f}'))
        others = g[~g.paper_model]
        row += [span(others.assign(p=others[f'shared_{TAU}'] * 100), 'p', '{:.1f}'), span(others, f'new_{TAU}'),
                span(others.assign(p=others[f'new_share_{TAU}'] * 100), 'p', '{:.1f}'),
                span(g, 'dup'), span(g.assign(p=g.dup_share * 100), 'p', '{:.1f}')]
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def figure(frame: pd.DataFrame, times: dict, outdir: Path, datasets=('macaque', 'aeon')) -> None:
    """Five-row, two-column appendix figure (dataset per column)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'axes.titlesize': 8})
    palette = ['#1b6ca8', '#d95f02', '#1b9e77', '#7570b3']
    fig, axes = plt.subplots(5, len(datasets), figsize=(5.5 if len(datasets) < 3 else 6.6, 8.0), sharex='col', squeeze=False,
                             constrained_layout=True)
    for c, dataset in enumerate(datasets):
        sub = frame[frame.dataset == dataset]
        sizes = sorted(sub.D.unique())
        x = {d: np.log2(d) for d in sizes}
        rng = np.random.default_rng(0)

        def pts(ax, column, color, label=None, hollow=None, **kw):
            """Plot individual runs (small), mean line (bold)."""
            jitter = lambda n: rng.uniform(-.04, .04, n)
            xs = np.array([x[d] for d in sub.D]) + jitter(len(sub))
            ax.scatter(xs, sub[column], s=7, color=color, alpha=.55, lw=0, zorder=2, **kw)
            means = sub.groupby('D')[column].mean()
            ax.plot([x[d] for d in means.index], means.values, '-o', color=color, ms=2.5, lw=1,
                    label=label, zorder=3)

        ax = axes[0, c]
        pts(ax, 'dead', palette[0])
        ax.set(ylabel='Dead latents', title=TITLES[dataset])
        ax = axes[1, c]
        t = times.get('aeon' if dataset == 'session_b' else dataset)
        pts(ax, 'train_seconds', '#bbbbbb')
        ax.lines[-1].set_visible(False)
        ax.collections[-1].set_alpha(.7)
        if t is not None and len(t):
            ax.plot([np.log2(d) for d in t.D], t.seconds, '-o', color=palette[1], ms=2.5, lw=1, zorder=4)
        ax.set(ylabel='Training time (s)')
        for r, rule, ylabel in ((2, 'auroc', 'AUROC of best latent'), (3, 'sel', 'Sel of best-AUROC latent')):
            ax = axes[r, c]
            if not FEATURES[dataset]:
                ax.axis('off')
                continue
            for k, name in enumerate(FEATURES[dataset]):
                col = f'{name}_auroc_{rule}'
                scored = sub
                means = scored.groupby('D')[col].mean()
                ax.scatter([x[d] for d in scored.D], scored[col], s=6, color=palette[k], alpha=.45, lw=0)
                ax.plot([x[d] for d in means.index], means.values, '-o', color=palette[k], ms=2.5, lw=1,
                        label=LABELS[dataset][k])
                paper = sub[sub.paper_model]
                ax.scatter([x[d] for d in paper.D], paper[col], s=22, facecolors='none', edgecolors='k', lw=.6, zorder=5)
            ax.set(ylabel=ylabel)
        ax = axes[4, c]
        big = sub[~sub.paper_model]
        for column, color, label in ((f'shared_{TAU}', palette[0], 'Paper-model latents reappearing'),
                                     (f'new_share_{TAU}', palette[1], 'Live latents without paper-model match'),
                                     ('dup_share', palette[2], 'Live latents with near-duplicate')):
            data = frame[frame.dataset == dataset]
            data = data[~data.paper_model] if column != 'dup_share' else data
            data = data.dropna(subset=[column])
            means = data.groupby('D')[column].mean() * 100
            ax.scatter([x[d] for d in data.D], data[column] * 100, s=6, color=color, alpha=.45, lw=0)
            ax.plot([x[d] for d in means.index], means.values, '-o', color=color, ms=2.5, lw=1, label=label)
        ax.set(ylabel='Share of latents (%)', xlabel='Dictionary size $D$')
        ax.set_xticks([x[d] for d in sizes], [str(d) for d in sizes])
        for ax, letter in zip(axes[:, c], 'abcde'):
            ax.grid(axis='y', color='.85', lw=.4)
            ax.set_axisbelow(True)
        axes[0, c].set_ylim(bottom=0)
        if FEATURES[dataset]:
            axes[2, c].set_ylim(.5, 1.02)
            axes[3, c].set_ylim(.4, 1.02)
        axes[4, c].set_ylim(-3, 103)
    axes[2, 0].legend(frameon=False, fontsize=5.5, loc='lower left', ncol=1)
    axes[4, -1].legend(frameon=False, fontsize=5.5, loc='center right')
    for i, ax in enumerate(axes[:, 0]):
        ax.text(-.32, 1.05, 'abcde'[i], transform=ax.transAxes, fontsize=9, fontweight='bold')
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(outdir / f'dictionary_size.{ext}', dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=CEPH / 'analysis')
    parser.add_argument('--figure-dir', type=Path, required=True)
    parser.add_argument('--tables', type=Path, default=CEPH / 'tables')
    parser.add_argument('--timing-dir', type=Path, default=CEPH / 'timing')
    args = parser.parse_args()
    frame = tidy(args.analysis)
    args.tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.tables / 'all_runs.csv', index=False)
    times = {d: timing(args.timing_dir / f'{d}.json', d) for d in ('macaque', 'aeon')}
    for d, t in times.items():
        t.to_csv(args.tables / f'timing_{d}.csv', index=False)
    for dataset in frame.dataset.unique():
        table = summary_table(frame, dataset)
        table.to_csv(args.tables / f'summary_{dataset}.csv', index=False)
        key = 'aeon' if dataset == 'session_b' else dataset
        (args.tables / f'summary_{dataset}.md').write_text(markdown(frame, dataset, times[key]) + '\n')
        print(dataset)
        print(table.round(3).to_string(index=False))
    figure(frame, times, args.figure_dir, tuple(d for d in ('macaque', 'aeon', 'session_b') if d in set(frame.dataset)))


if __name__ == '__main__':
    main()
