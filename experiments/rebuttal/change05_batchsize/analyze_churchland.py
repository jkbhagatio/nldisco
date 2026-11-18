"""Score the Section 4.2 models retrained at several batch sizes (development windows).

For each model: paper selection rule (best eligible AUROC and best eligible Sel, z>0 active)
for the four feature descriptions, dead latents, reconstruction variance explained, and the
share of the paper model's 192 latents that reappear (max correlation >= MATCH_THRESHOLD).
Run from the worktree root: python -m experiments.rebuttal.change05_batchsize.analyze_churchland
"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.churchland.sweeps.s10_endpoint_v1 import comparison_features as shared
from experiments.churchland.sweeps.s10_endpoint_v1 import feature_decoding as decoder
from experiments.churchland.sweeps.s10_endpoint_v1 import nice_cutoffs_revision as revision
from experiments.rebuttal.change05_batchsize import scoring

MAIN = Path('/nfs/nhome/live/jbhagat/nldisco')
SWEEP = Path('/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1')
SEEDS_FOLDER = SWEEP / 'width192_checks/nldisco_seeds_1_to_5'
REFERENCE = SEEDS_FOLDER / 'd192_k16_l2_seed1'
FROZEN = MAIN / 'experiments/churchland/outputs/s10_endpoint_revision_v4_nice_cutoffs/frozen_spec.json'
RUNS = Path('/ceph/aeon/aeon/nldisco/rebuttal_20261118/change05_batchsize/churchland')
RESULTS = Path(__file__).resolve().parent / 'results'
FEATURES = shared.EXPECTED_FEATURES


def load_features() -> dict:
    """Current (v4) feature definitions on development windows; recording read from the main checkout."""
    shared.ROOT = MAIN
    decoder.DATA = MAIN / 'experiments/churchland/data/nitschke_20090812'
    return revision.features()


def score_model(activations: np.ndarray, features: dict) -> dict:
    """Best-Sel and best-AUROC eligible latent per feature."""
    result = {}
    for name in FEATURES:
        feature = features[name]
        scores = scoring.score_latents(activations[feature.rows], feature.labels, feature.trials)
        entry = {}
        for criterion in ('sel', 'auroc'):
            latent = scoring.best_latent(scores, criterion)
            entry[criterion] = None if latent is None else dict(
                latent=latent, sel=float(scores['sel'][latent]), auroc=float(scores['auroc'][latent]),
                tpr=float(scores['tpr'][latent]), fpr=float(scores['fpr'][latent]))
        entry['n_eligible'] = int(scores['eligible'].sum())
        result[name] = entry
    return result


def matched_scores(activations: np.ndarray, features: dict, paper_latents: list, matches: dict) -> list:
    """Score the retrained latent matched to each latent the paper displays."""
    rows = []
    for item in paper_latents:
        feature = features[item['feature']]
        partner = int(matches['index'][item['latent']])
        scores = scoring.score_latents(activations[feature.rows][:, [partner]], feature.labels, feature.trials)
        rows.append(dict(item, partner=partner, correlation=float(matches['correlation'][item['latent']]),
                         partner_sel=float(scores['sel'][0]), partner_auroc=float(scores['auroc'][0])))
    return rows


def health(folder: Path, variance: np.ndarray) -> dict:
    """Dead latents and variance explained of the 192-latent level from a run summary."""
    level = json.loads((folder / 'summary.json').read_text())['levels']['192']
    mse = np.asarray(level['mse_by_bin'])
    return dict(dead_latents=int(round(level['dead_feature_fraction'] * 192)), mean_l0=level['mean_l0'],
                mse=float(mse.mean()), variance_explained=scoring.variance_explained(mse, variance))


def model_list() -> list:
    """(label, group, batch size, seed, folder): the paper model, its sibling seeds, new runs."""
    models = [('paper_seed1', 'paper', 1024, 1, REFERENCE)]
    models += [(f'paper_seed{s}', 'paper_seeds', 1024, s, SEEDS_FOLDER / f'd192_k16_l2_seed{s}') for s in (2, 3, 4, 5)]
    models += [('check_seed1_bs1024', 'check', 1024, 1, RUNS / 'check_seed1_bs1024')]
    for batch_size in (512, 1024, 2048):
        for seed in (11, 12, 13):
            models.append((f'bs{batch_size}_seed{seed}', 'retrained', batch_size, seed,
                           RUNS / f'bs{batch_size}_seed{seed}'))
    return [m for m in models if (m[4] / 'activations_level_192.npy').exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    features = load_features()
    reference = np.load(REFERENCE / 'activations_level_192.npy', mmap_mode='r')
    frozen = json.loads(FROZEN.read_text())['selections']
    paper_latents = sorted({(s['feature'], s['criterion'], s['latent_id']) for s in frozen if s['method'] == 'nldisco'})
    paper_latents = [dict(feature=f, criterion=c, latent=l) for f, c, l in paper_latents]
    variance = np.asarray(json.loads((RUNS / 'check_seed1_bs1024/summary.json').read_text())
                          ['levels']['192']['variance_by_bin'])
    output = {}
    for label, group, batch_size, seed, folder in model_list():
        activations = np.load(folder / 'activations_level_192.npy', mmap_mode='r')
        index = np.load(folder / 'activation_index.npz')
        np.testing.assert_array_equal(index['source_index'], np.load(SWEEP / 'shared/index.npz')['anchor_index'])
        activations = np.asarray(activations)
        matches = scoring.match_latents(scoring.column_correlation(np.asarray(reference), activations,
                                                                   device=args.device))
        output[label] = dict(group=group, batch_size=batch_size, seed=seed,
                             features=score_model(activations, features),
                             health=health(folder, variance),
                             match_share=matches['share'], match_median_correlation=matches['median_correlation'],
                             matched_paper_latents=matched_scores(activations, features, paper_latents, matches))
        print(label, 'share', round(matches['share'], 3), json.dumps(output[label]['health']), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'churchland_models.json').write_text(json.dumps(
        dict(match_threshold=scoring.MATCH_THRESHOLD, paper_latents=paper_latents, models=output), indent=1))


if __name__ == '__main__':
    main()
