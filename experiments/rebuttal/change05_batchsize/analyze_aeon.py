"""Score the Section 4.3 (Aeon) D=192, k=12 model retrained at several batch sizes.

Wheel-activity feature (endpoint wheel-rim speed > 0.75 cm/s, complete wheel support), all
windows, native z>0 activity. Same selection rule as the macaque analysis: best eligible
AUROC and best eligible Sel (coverage >= 5%); the paper displays latent 52.
Run from the worktree root: python -m experiments.rebuttal.change05_batchsize.analyze_aeon
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.aeon.sweep_20260921 import paper_results
from experiments.rebuttal.change05_batchsize import scoring

ROOT = Path('/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20')
REFERENCE = ROOT / 'd192/k12_seed0'
RUNS = Path('/ceph/aeon/aeon/nldisco/rebuttal_20261118/change05_batchsize/aeon')
RESULTS = Path(__file__).resolve().parent / 'results'
PAPER_LATENT = 52


def health(folder: Path) -> dict:
    """Dead latents and variance explained (summed over the 20 lags) at the 192 level."""
    metrics = json.loads((folder / 'metrics.json').read_text())
    lags = pd.read_csv(folder / 'reconstruction_by_lag.csv')
    lags = lags[lags.level == 192]
    variance = lags.mse / (1. - lags.variance_weighted_r2)
    return dict(dead_latents=len(metrics['dead_latents']), mean_l0=metrics['mean_l0'], mse=metrics['mse']['192'],
                variance_explained=scoring.variance_explained(lags.mse.to_numpy(), variance.to_numpy()))


def model_list() -> list:
    models = [('paper_seed0', 'paper', 1024, 0, REFERENCE)]
    for batch_size in (512, 1024, 2048):
        for seed in (11, 12, 13):
            models.append((f'bs{batch_size}_seed{seed}', 'retrained', batch_size, seed, RUNS / f'bs{batch_size}_seed{seed}'))
    return [m for m in models if (m[4] / 'COMPLETE.json').exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    behavior = pd.read_parquet(ROOT / 'data/behavior.parquet')
    endpoints = np.load(REFERENCE / 'endpoint_bins.npy')
    conditions, _ = paper_results.definitions(behavior, endpoints)
    condition, valid = conditions['wheel']
    reference = np.load(REFERENCE / 'activations.npy', mmap_mode='r')
    output = {}
    for label, group, batch_size, seed, folder in model_list():
        np.testing.assert_array_equal(np.load(folder / 'endpoint_bins.npy'), endpoints)
        activations = np.load(folder / 'activations.npy', mmap_mode='r')
        scores = scoring.score_latents(np.asarray(activations[valid]), condition[valid], device=args.device)
        features = dict(n_eligible=int(scores['eligible'].sum()))
        for criterion in ('sel', 'auroc'):
            latent = scoring.best_latent(scores, criterion)
            features[criterion] = dict(latent=latent, sel=float(scores['sel'][latent]), auroc=float(scores['auroc'][latent]),
                                       tpr=float(scores['tpr'][latent]), fpr=float(scores['fpr'][latent]))
        matches = scoring.match_latents(scoring.column_correlation(reference, activations, device=args.device))
        partner = int(matches['index'][PAPER_LATENT])
        output[label] = dict(group=group, batch_size=batch_size, seed=seed, features=dict(wheel=features),
                             health=health(folder), match_share=matches['share'],
                             match_median_correlation=matches['median_correlation'],
                             matched_paper_latents=[dict(
                                 feature='wheel', criterion='paper', latent=PAPER_LATENT, partner=partner,
                                 correlation=float(matches['correlation'][PAPER_LATENT]),
                                 partner_sel=float(scores['sel'][partner]), partner_auroc=float(scores['auroc'][partner]))])
        print(label, json.dumps(features), json.dumps(output[label]['health']), 'share', round(matches['share'], 3), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'aeon_models.json').write_text(json.dumps(
        dict(match_threshold=scoring.MATCH_THRESHOLD, models=output), indent=1))


if __name__ == '__main__':
    main()
