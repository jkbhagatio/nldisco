"""Activation-threshold sensitivity of coverage, specificity and selectivity (Section 4.2).

The latents selected in the paper stay fixed (index and sign). Each is re-scored on the
development windows while the activation rate that sets its threshold varies:

* sweep: threshold = ``1 - rate`` quantile of the sign-oriented activation (the paper's rule),
* paper: the paper's rate (dense baselines) or the native rule ``z > 0`` (NLDisco),
* half / double: half and twice the paper's rate,
* best: the threshold that maximises Sel among thresholds satisfying the paper's eligibility
  rule (Cov >= 0.05 and activity in >= 20 positive trials).

Everything is computed from saved representations; nothing is retrained. AUROC is computed
from raw activations and does not depend on the threshold.

Adding a method is one entry in ``METHODS`` (or ``--extra key:label:path:color`` on the command
line). Methods without selections in ``frozen_spec.json`` get their best-Sel and best-AUROC
latent from the paper's own selection rule.

Run from the worktree root::

    $PY -m experiments.rebuttal.change09_thresholds.threshold_sensitivity --out <dir>
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from beartype import beartype
from scipy.stats import rankdata

from experiments.churchland.sweeps.s10_endpoint_v1 import comparison_features as shared
from experiments.churchland.sweeps.s10_endpoint_v1 import feature_decoding as decoder
from experiments.churchland.sweeps.s10_endpoint_v1 import nice_cutoffs_revision as revision

SOURCE = Path(shared.ROOT) / 'experiments/churchland/outputs/s10_endpoint_revision_v4_nice_cutoffs'
MIN_COVERAGE, MIN_TRIALS = 0.05, 20  # the paper's eligibility rule
FEATURE_ORDER = ('recent_braking', 'fast_target_specific', 'early_leftward', 'late_target_specific')
SETTINGS = ('half', 'paper', 'double', 'best')


@dataclass(frozen=True)
class MethodConfig:
    """One compared method; ``representation=None`` takes the path from ``frozen_spec.json``.

    Attributes:
        key: Method key used in ``frozen_spec.json`` and the outputs.
        label: Name used in figure and table.
        color: Matplotlib color.
        dense: Dense coordinates (rate-calibrated thresholds) if True; sparse latent with
            native threshold ``z > 0`` if False.
        representation: Optional ``{'path': ...}`` record of an exported representation.
        selections: ``'frozen'`` reads ``frozen_spec.json``; ``'paper_rule'`` re-applies the
            paper's best-Sel and best-AUROC rule to the representation.
    """

    key: str
    label: str
    color: str
    dense: bool = True
    representation: Optional[dict] = None
    selections: str = 'frozen'


METHODS = (
    MethodConfig('nldisco', 'NLDisco', '#1f77b4', dense=False),
    MethodConfig('cebra', 'CEBRA-Time', '#ff7f0e'),
    MethodConfig('langevinflow', 'LangevinFlow', '#2ca02c'),
    # ST-NDT: add `MethodConfig('stndt', 'ST-NDT', '#9467bd', representation={'path': ...},
    # selections='paper_rule')` or pass `--extra stndt:ST-NDT:<path>:#9467bd`.
)


@beartype
def oriented_scores(x: np.ndarray, y: np.ndarray, threshold: float) -> tuple:
    """Return ``(tpr, fpr)`` of the rule ``x > threshold`` on boolean labels ``y``."""
    active = x > threshold
    return float(active[y].mean()), float(active[~y].mean())


@beartype
def selectivity(tpr: float, fpr: float) -> float:
    """Sel = TPR / (TPR + FPR), zero when the latent never fires."""
    return tpr / (tpr + fpr) if tpr + fpr > 0 else 0.0


@beartype
def rate_threshold(x: np.ndarray, rate: float) -> float:
    """The paper's rule: the ``1 - rate`` quantile, so about ``rate`` of windows are active."""
    return float(np.quantile(x, 1 - rate))


@beartype
def score_at_rate(x: np.ndarray, y: np.ndarray, rate: float) -> dict:
    """Coverage, specificity, selectivity and achieved rate of the rate-calibrated threshold."""
    threshold = rate_threshold(x, rate)
    tpr, fpr = oriented_scores(x, y, threshold)
    return dict(requested_rate=rate, achieved_rate=float((x > threshold).mean()),
                threshold=threshold, cov=tpr, spec=1 - fpr, sel=selectivity(tpr, fpr))


@beartype
def rate_curve(x: np.ndarray, y: np.ndarray, rates: np.ndarray) -> list:
    """Scores at each requested activation rate."""
    return [score_at_rate(x, y, float(r)) for r in rates]


@beartype
def best_threshold(x: np.ndarray, y: np.ndarray, trials: np.ndarray,
                   min_coverage: float = MIN_COVERAGE) -> dict:
    """Exact threshold maximising Sel subject to the paper's eligibility rule.

    Candidates are all distinct activation values (the rule is ``x > t``). Ties in Sel are
    broken towards higher coverage.

    Args:
        x: Sign-oriented activations on the development windows.
        y: Boolean condition labels.
        trials: Trial id per window (coverage must reach >= 20 positive trials).
        min_coverage: Smallest admissible coverage (the paper's 0.05 by default).

    Returns:
        Scores at the best threshold (``cov``, ``spec``, ``sel``, ``threshold``,
        ``achieved_rate``); ``eligible`` is False if no threshold satisfies the rule.
    """
    pos, neg = np.sort(x[y]), np.sort(x[~y])
    candidates = np.unique(x)
    tpr = 1 - np.searchsorted(pos, candidates, side='right') / len(pos)
    fpr = 1 - np.searchsorted(neg, candidates, side='right') / len(neg)
    trial_ids, inverse = np.unique(trials[y], return_inverse=True)
    trial_max = np.full(len(trial_ids), -np.inf)
    np.maximum.at(trial_max, inverse, x[y])
    trial_max.sort()
    support = len(trial_max) - np.searchsorted(trial_max, candidates, side='right')
    ok = (tpr >= min_coverage - 1e-12) & (support >= MIN_TRIALS)
    if not ok.any():
        return dict(eligible=False)
    sel = np.divide(tpr, tpr + fpr, out=np.zeros_like(tpr), where=tpr + fpr > 0)
    index = np.flatnonzero(ok)
    k = index[np.lexsort((tpr[index], sel[index]))[-1]]
    t = float(candidates[k])
    return dict(eligible=True, threshold=t, achieved_rate=float((x > t).mean()),
                cov=float(tpr[k]), spec=float(1 - fpr[k]), sel=float(sel[k]))


@beartype
def orient(values: np.ndarray, rows: np.ndarray, latent: int, orientation: int) -> np.ndarray:
    """Development-window activations of one latent with the paper's sign."""
    return np.asarray(values[rows, latent], dtype=np.float64) * orientation


@beartype
def auroc(x: np.ndarray, y: np.ndarray) -> float:
    """Threshold-free AUROC (ties count one half), orientation already applied."""
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    ranks = rankdata(x)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@beartype
def settings_for(x: np.ndarray, feature: shared.FeatureData, paper_rate: float,
                 dense: bool, matched_coverage: Optional[float] = None) -> dict:
    """Scores at half, paper, twice the paper's rate, and own best threshold.

    For a sparse latent the paper setting is its native rule ``z > 0``; twice that rate is not
    reachable (no further nonzero activations), so ``double`` is None.
    """
    y = feature.labels.astype(bool)
    if dense:
        out = {name: score_at_rate(x, y, paper_rate * m)
               for name, m in (('half', .5), ('paper', 1.), ('double', 2.))}
    else:
        tpr, fpr = oriented_scores(x, y, 0.0)
        native = dict(requested_rate=paper_rate, achieved_rate=float((x > 0).mean()),
                      threshold=0.0, cov=tpr, spec=1 - fpr, sel=selectivity(tpr, fpr))
        out = dict(paper=native, half=score_at_rate(x, y, paper_rate / 2), double=None)
    out['best'] = best_threshold(x, y, feature.trials)
    if matched_coverage is not None:
        out['matched'] = best_threshold(x, y, feature.trials, max(matched_coverage, MIN_COVERAGE))
    return out


@beartype
def log_rate_grid(paper_rate: float, n: int = 61) -> np.ndarray:
    """Log-spaced rates from 0.1% to 60%, including half, one and two times the paper's."""
    grid = np.geomspace(0.001, 0.6, n)
    return np.unique(np.r_[grid, paper_rate / 2, paper_rate, min(2 * paper_rate, .99)])


@beartype
def paper_rule_selection(values: np.ndarray, features: dict, method: str) -> list:
    """Best-Sel and best-AUROC eligible latent per feature under the paper's rule."""
    rows = []
    for name in FEATURE_ORDER:
        scored = shared.score_dense(values, features[name])
        for criterion in ('sel', 'auroc'):
            pick = max((r for r in scored if r['eligible']),
                       key=lambda r: (r[criterion], -r['latent_id']))
            rows.append(dict(method=method, feature=name, criterion=criterion,
                             latent_id=pick['latent_id'], orientation=pick['orientation'],
                             threshold=pick['threshold'], achieved_rate=pick['achieved_rate'],
                             tpr=pick['tpr'], fpr=pick['fpr'], sel=pick['sel'],
                             auroc=pick['auroc']))
    return rows


@beartype
def selected_latents(spec: dict, config: MethodConfig, values: np.ndarray,
                     features: dict) -> list:
    """Latents to re-score, as listed in Table 1: NLDisco's illustrative latent for the two
    main-text features and its best-Sel and best-AUROC latents for the two supplementary ones;
    each baseline's best-Sel and best-AUROC latent (identical picks are merged)."""
    if config.selections == 'paper_rule':
        rows = paper_rule_selection(values, features, config.key)
    else:
        rows = [r for r in spec['selections'] if r['method'] == config.key]
    seen = {}
    for row in rows:
        main = row['feature'] in spec['main_features']
        if not config.dense and (row['criterion'] == 'illustrative') != main:
            continue  # NLDisco as in Table 1: illustrative latent (main) or best-Sel/AUROC
        key = (row['feature'], row['latent_id'])
        if key in seen:
            seen[key]['selectors'].append(row['criterion'])
        else:
            seen[key] = dict(row, selectors=[row['criterion']])
    return list(seen.values())


@beartype
def analyse(configs: tuple, features: dict, spec: dict) -> dict:
    """Run the full sensitivity analysis; returns a JSON-serialisable dict."""
    results, checks, reference_cov = [], [], {}
    for config in configs:
        record = config.representation or spec['representations'][config.key]
        values = decoder.load_representation(record, decoder.SWEEP)
        for row in selected_latents(spec, config, values, features):
            feature = features[row['feature']]
            y = feature.labels.astype(bool)
            x = orient(values, feature.rows, row['latent_id'], row['orientation'])
            native = float((x > 0).mean())
            paper_rate = feature.reference_rate if config.dense else native
            curve = rate_curve(x, y, log_rate_grid(paper_rate) if config.dense
                               else np.unique(np.r_[np.geomspace(0.001, native, 41)[:-1],
                                                    paper_rate / 2]))
            if not config.dense:  # native operating point closes the curve
                tpr, fpr = oriented_scores(x, y, 0.0)
                curve.append(dict(requested_rate=native, achieved_rate=native, threshold=0.0,
                                  cov=tpr, spec=1 - fpr, sel=selectivity(tpr, fpr)))
            setting = settings_for(x, feature, paper_rate, config.dense,
                                   reference_cov.get(row['feature']) if config.dense else None)
            if not config.dense and row['latent_id'] == spec['current_rate_reference_latents'][
                    row['feature']]:
                reference_cov[row['feature']] = setting['paper']['cov']
            results.append(dict(
                method=config.key, label=config.label, feature=row['feature'],
                latent_id=row['latent_id'], orientation=row['orientation'],
                selectors=row['selectors'], dense=config.dense, paper_rate=paper_rate,
                auroc=auroc(x, y), prevalence=float(y.mean()), n_windows=int(len(y)),
                curve=curve, settings=setting))
            if config.selections == 'frozen' and 'tpr' in row:  # reproduce the paper's scores
                paper = setting['paper']
                checks.append(dict(method=config.key, feature=row['feature'],
                                   latent_id=row['latent_id'], selectors=row['selectors'],
                                   max_abs_diff=max(abs(paper['cov'] - row['tpr']),
                                                    abs(paper['spec'] - (1 - row['fpr'])),
                                                    abs(paper['sel'] - row['sel']),
                                                    abs(results[-1]['auroc'] - row['auroc']))))
    return dict(results=results, reproduction_checks=checks)


@beartype
def load_features() -> dict:
    """The four development features with the current (v4) labels and the paper's rates."""
    features = revision.features()
    spec = json.loads((SOURCE / 'frozen_spec.json').read_text())
    for name, rate in spec['dense_reference_rates'].items():
        assert abs(features[name].reference_rate - rate) < 1e-12, name
    return features


@beartype
def parse_extra(text: str) -> MethodConfig:
    """``key:label:path:color`` -> a method scored with the paper's selection rule."""
    key, label, path, color = text.split(':', 3)
    return MethodConfig(key, label, color, representation=dict(path=path), selections='paper_rule')


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', type=Path, required=True, help='New output directory.')
    parser.add_argument('--extra', action='append', default=[], help='key:label:path:color')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((SOURCE / 'frozen_spec.json').read_text())
    configs = METHODS + tuple(parse_extra(e) for e in args.extra)
    report = analyse(configs, load_features(), spec)
    report['methods'] = [dict(key=c.key, label=c.label, color=c.color, dense=c.dense)
                         for c in configs]
    report['paper_rates'] = spec['dense_reference_rates']
    (args.out / 'threshold_sensitivity.json').write_text(json.dumps(report, indent=1) + '\n')
    worst = max(c['max_abs_diff'] for c in report['reproduction_checks'])
    print(f'wrote {args.out}; max |re-score - paper| over {len(report["reproduction_checks"])} '
          f'latents = {worst:.2e}')


if __name__ == '__main__':
    main()
