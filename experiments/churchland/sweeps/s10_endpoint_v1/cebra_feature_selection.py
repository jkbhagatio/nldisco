"""Preflight and shared-protocol selection of one frozen CEBRA configuration."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from beartype import beartype

ROOT = Path('/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1')
FEATURES = ('recent_braking', 'early_leftward', 'fast_target_specific', 'late_target_specific')


@beartype
def sha256(path: Path) -> str:
    """Hash an immutable input without holding the entire file in memory."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@beartype
def load_export(path: Path, expected_index: np.ndarray, dimension: int) -> np.ndarray:
    """Validate an existing CEBRA export against the shared endpoint order."""
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved['source_index'], expected_index)
        values = saved['activations']
    if values.shape != (len(expected_index), dimension):
        raise ValueError(f'Unexpected embedding shape {values.shape} at {path}')
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError(f'Expected finite float32 embeddings at {path}')
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1., atol=1e-5)
    return values


@beartype
def inventory(root: Path) -> list[dict]:
    """Audit the complete approved twelve-configuration grid without labels."""
    with np.load(root / 'shared/index.npz') as saved:
        expected_index = saved['anchor_index']
    if len(expected_index) != 130507:
        raise ValueError('Expected 130507 common endpoint windows.')
    rows = []
    for dimension, temperature, lag in itertools.product((32, 64, 128), (.3, 1.), (1, 5)):
        config_id = f"d{dimension}_t{str(temperature).replace('.', 'p')}_lag{lag}"
        folder = root / 'cebra' / config_id
        config = json.loads((folder / 'config.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        for key, value in dict(dimension=dimension, temperature=temperature, lag=lag, seed=0).items():
            if config[key] != value:
                raise ValueError(f'Unexpected {key} for {config_id}')
        if summary['status'] != 'complete' or summary['reload_max_abs_error'] != 0:
            raise ValueError(f'Incomplete or inconsistent checkpoint for {config_id}')
        path = folder / 'endpoint_activations.npz'
        values = load_export(path, expected_index, dimension)
        if sha256(folder / 'cebra_source.py') != config['source_sha256']:
            raise ValueError(f'Source hash mismatch for {config_id}')
        rows.append(dict(config_id=config_id, dimension=dimension, temperature=temperature, lag=lag,
                         export_path=str(path), shape=list(values.shape),
                         export_sha256=sha256(path), config_sha256=sha256(folder / 'config.json'),
                         training_source_sha256=config['source_sha256']))
        print(f'Validated {config_id}: {values.shape}', flush=True)
    return rows


@beartype
def choose_configuration(configurations: list[dict], feature_names: tuple) -> dict:
    """Select one complete configuration by the equally weighted four-feature mean."""
    if tuple(feature_names) != FEATURES:
        raise ValueError('All four finalized features are required in frozen order.')
    complete = []
    for config in configurations:
        choices = config['feature_choices']
        if set(choices) != set(FEATURES):
            raise ValueError('Incomplete feature set; selection is forbidden.')
        if all(choices[name] is not None for name in FEATURES):
            mean = float(np.mean([choices[name]['auroc'] for name in FEATURES]))
            complete.append(dict(config, mean_auroc=mean))
    if not complete:
        raise ValueError('No configuration has an eligible dimension for all four features.')
    return max(complete, key=lambda config: config['mean_auroc'])


@beartype
def select(root: Path) -> None:
    """Use only the shared frozen feature and scoring contract for numerical selection."""
    from experiments.churchland.sweeps.s10_endpoint_v1 import comparison_features as shared

    features = shared.load_development_features()
    if set(features) != set(FEATURES) or tuple(shared.EXPECTED_FEATURES) != FEATURES:
        raise RuntimeError('Selection blocked: all four finalized feature definitions, including 181, are required.')
    destination = root / 'feature_comparison' / 'cebra'
    output = destination / 'selection.json'
    if output.exists():
        raise FileExistsError(output)
    preflight = json.loads((destination / 'preflight.json').read_text())
    if sha256(root / 'shared/index.npz') != preflight['shared_index_sha256']:
        raise ValueError('Shared endpoint index changed after preflight.')
    with np.load(root / 'shared/index.npz') as index:
        anchors = index['anchor_index']
    per_config, all_candidates = [], []
    for item in preflight['exports']:
        path = Path(item['export_path'])
        if sha256(path) != item['export_sha256']:
            raise ValueError(f'Export changed after preflight: {path}')
        values = load_export(path, anchors, item['dimension'])
        choices = {}
        for name in FEATURES:
            rows = shared.score_dense(values, features[name])
            choices[name] = shared.best_eligible(rows)
            all_candidates.extend(dict(config_id=item['config_id'], feature=name, **row) for row in rows)
        complete = all(choices[name] is not None for name in FEATURES)
        per_config.append(dict(config_id=item['config_id'], feature_choices=choices,
                               eligible_for_selection=complete,
                               mean_auroc=float(np.mean([choices[name]['auroc'] for name in FEATURES])) if complete else None))
        print(json.dumps(per_config[-1]), flush=True)
    winner = choose_configuration(per_config, FEATURES)
    definitions = {name: dict(definition=features[name].definition,
                             reference_activity_rate=features[name].reference_rate,
                             valid_development_rows=len(features[name].rows),
                             rows_sha256=hashlib.sha256(np.asarray(features[name].rows).tobytes()).hexdigest(),
                             labels_sha256=hashlib.sha256(np.asarray(features[name].labels).tobytes()).hexdigest(),
                             trials_sha256=hashlib.sha256(np.asarray(features[name].trials).tobytes()).hexdigest())
                   for name in FEATURES}
    report = dict(status='complete', method='CEBRA', selected_configuration=winner,
                  selection_rule='Single configuration maximizing the equal-weight mean of four best eligible positive-oriented AUROCs.',
                  tie_rule='First in fixed preflight grid order if configuration means tie exactly.',
                  dimension_rule='Shared best_eligible AUROC choice; these exact dimensions are frozen for illustration and eventual single-latent decoders.',
                  development_only=True, test_labels_accessed=False, decoder_fitted=False,
                  features=definitions, configurations=per_config,
                  comparison_source_sha256=sha256(Path(shared.__file__)),
                  evaluator_source_sha256=sha256(Path(__file__)),
                  preflight_sha256=sha256(destination / 'preflight.json'))
    (destination / 'all_candidates.json').write_text(json.dumps(all_candidates, indent=2))
    output.write_text(json.dumps(report, indent=2))
    (destination / 'selection_source.py').write_bytes(Path(__file__).read_bytes())
    (destination / 'comparison_features_source.py').write_bytes(Path(shared.__file__).read_bytes())
    print(json.dumps(dict(selected_configuration=winner, output=str(output)), indent=2))


@beartype
def sensitivity(root: Path) -> None:
    """Verify frozen dimensions and assess 100 fixed-size candidate subsets."""
    from sklearn.metrics import roc_auc_score

    from experiments.churchland.sweeps.s10_endpoint_v1 import comparison_features as shared

    destination = root / 'feature_comparison' / 'cebra'
    output = destination / 'candidate_count_sensitivity.json'
    if output.exists():
        raise FileExistsError(output)
    selection = json.loads((destination / 'selection.json').read_text())
    if sha256(Path(shared.__file__)) != selection['comparison_source_sha256']:
        raise ValueError('The shared scoring contract changed after selection.')
    winner = selection['selected_configuration']
    folder = root / 'cebra' / winner['config_id']
    config = json.loads((folder / 'config.json').read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    features = shared.load_development_features()
    with np.load(root / 'shared/index.npz') as saved:
        anchors = saved['anchor_index']
    values = load_export(folder / 'endpoint_activations.npz', anchors, config['dimension'])
    verification = {}
    for name in FEATURES:
        chosen = winner['feature_choices'][name]
        feature = features[name]
        x = values[feature.rows, chosen['latent_id']] * chosen['orientation']
        measured_auc = float(roc_auc_score(feature.labels, x))
        measured_threshold = float(np.quantile(x, 1 - feature.reference_rate))
        measured_rate = float((x > chosen['threshold']).mean())
        np.testing.assert_allclose(measured_auc, chosen['auroc'], atol=1e-12)
        np.testing.assert_allclose(measured_threshold, chosen['threshold'], atol=1e-8)
        np.testing.assert_allclose(measured_rate, chosen['achieved_rate'], atol=1e-12)
        verification[name] = dict(sklearn_auroc=measured_auc, threshold=measured_threshold,
                                  achieved_rate=measured_rate,
                                  rate_error=measured_rate - feature.reference_rate)
    all_candidates = json.loads((destination / 'all_candidates.json').read_text())
    candidates = {name: [row for row in all_candidates
                        if row['config_id'] == winner['config_id'] and row['feature'] == name]
                  for name in FEATURES}
    generator = np.random.default_rng(0)
    draws = []
    for draw in range(100):
        subset = generator.choice(config['dimension'], size=32, replace=False)
        subset_set = set(subset.tolist())
        choices = {name: shared.best_eligible([row for row in candidates[name]
                                              if row['latent_id'] in subset_set]) for name in FEATURES}
        complete = all(value is not None for value in choices.values())
        draws.append(dict(draw=draw, candidate_ids=sorted(subset_set), feature_choices=choices,
                          eligible_all_four=complete,
                          mean_auroc=shared.configuration_score(choices) if complete else None))
    def distribution(numbers: list) -> dict:
        return dict(n=len(numbers), minimum=float(np.min(numbers)), mean=float(np.mean(numbers)),
                    q05=float(np.quantile(numbers, .05)), median=float(np.median(numbers)),
                    q95=float(np.quantile(numbers, .95)), maximum=float(np.max(numbers))) if numbers else dict(n=0)
    complete_draws = [draw for draw in draws if draw['eligible_all_four']]
    report = dict(status='complete', selected_config_id=winner['config_id'], seed=0,
                  candidate_count=32, original_candidate_count=config['dimension'], repetitions=100,
                  protocol='Without-replacement subsets of the same 32 dimensions for all four features in each draw; fixed selected model, existing shared eligibility and best-AUROC rules. No model reselection.',
                  complete_draws=len(complete_draws),
                  mean_auroc_distribution=distribution([draw['mean_auroc'] for draw in complete_draws]),
                  per_feature={name: {metric: distribution([draw['feature_choices'][name][metric]
                                        for draw in draws if draw['feature_choices'][name] is not None])
                                      for metric in ('auroc', 'sel')} for name in FEATURES},
                  frozen_dimension_verification=verification, selected_training_config=config,
                  selected_training_summary=summary, draws=draws,
                  selection_sha256=sha256(destination / 'selection.json'),
                  comparison_source_sha256=sha256(Path(shared.__file__)))
    output.write_text(json.dumps(report, indent=2))
    (destination / 'sensitivity_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: value for key, value in report.items() if key not in ('draws', 'selected_training_config')}, indent=2))


@beartype
def main() -> None:
    """Write label-free preflight while the frozen four-feature contract is pending."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--select', action='store_true')
    parser.add_argument('--sensitivity', action='store_true')
    args = parser.parse_args()
    if args.sensitivity:
        sensitivity(args.root)
        return
    if args.select:
        select(args.root)
        return
    if not args.preflight:
        raise RuntimeError('Numerical selection is gated on the finalized shared four-feature contract, including feature 181.')
    rows = inventory(args.root)
    destination = args.root / 'feature_comparison' / 'cebra'
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / 'preflight.json'
    if output.exists():
        raise FileExistsError(output)
    report = dict(status='preflight_complete_selection_pending', method='CEBRA', configurations=len(rows),
                  blocker='Feature 181 definition pending; no label scoring or final configuration selection performed.',
                  planned_selection='One configuration maximizing equal-weight mean of four best eligible feature AUROCs; no per-feature configuration oracle.',
                  feature_latent_ids=[31, 132, 181, 4],
                  threshold_protocol='Orient dense dimensions by positive development AUROC; strict > threshold matching the fixed NLDisco feature activity rate without splitting ties.',
                  split='Numerical scoring will use development split !=2 only; preflight reads no behavioral metadata.',
                  shared_index_sha256=sha256(args.root / 'shared/index.npz'), exports=rows,
                  evaluator_source_sha256=sha256(Path(__file__)))
    output.write_text(json.dumps(report, indent=2))
    (destination / 'preflight_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status=report['status'], configurations=len(rows), output=str(output))))


if __name__ == '__main__':
    main()
