"""Fit a chronological single-latent decoder for the existing Aeon Figure 5 feature."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.aeon.sweep_20260921.shared_analysis import complete_support

ROOT = Path('/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20')
GRID = (.001, .01, .1, 1., 10., 100., 1000.)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / 'fit_decoder.py')
    arrays_path = ROOT / 'paper_revision_wheelcms_20260921/feature_arrays.npz'
    with np.load(arrays_path, allow_pickle=False) as arrays:
        endpoints = arrays['endpoint_bins']
        values = arrays['values_wheel'].astype(np.float64)
        labels = arrays['condition_wheel']
        valid = arrays['valid_wheel']
    behavior = pd.read_parquet(ROOT / 'data/behavior.parquet', columns=['wheel_speed', 'wheel_valid'])
    speed = behavior.wheel_speed.to_numpy(dtype=float)
    n_bins = len(behavior)
    assert n_bins == 3240000
    np.testing.assert_array_equal(labels, speed[endpoints] > .75 / (8 * np.pi))
    expected_valid = complete_support(behavior.wheel_valid.to_numpy() & np.isfinite(speed))[endpoints]
    np.testing.assert_array_equal(valid, expected_valid)
    assert np.isfinite(values).all() and (np.diff(endpoints) > 0).all()

    # Boundaries are defined in physical recording time, before filtering invalid rows.
    bounds = (0, n_bins * 8 // 10, n_bins * 9 // 10, n_bins)
    masks = {name: valid & (endpoints - 19 >= left) & (endpoints < right)
             for name, left, right in zip(('train', 'validation', 'test'), bounds[:-1], bounds[1:])}
    for name, mask in masks.items():
        assert mask.any() and np.unique(labels[mask]).size == 2, name
    for earlier, later in [('train', 'validation'), ('validation', 'test')]:
        assert not (masks[earlier] & masks[later]).any()
        assert endpoints[masks[earlier]].max() < (endpoints[masks[later]] - 19).min()
    support = {name: dict(windows=int(mask.sum()), positives=int(labels[mask].sum()),
                         negatives=int((~labels[mask]).sum()),
                         first_endpoint_bin=int(endpoints[mask][0]),
                         last_endpoint_bin=int(endpoints[mask][-1]))
               for name, mask in masks.items()}
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    model='d192/k12_seed0', latent=52, input='continuous sparse latent amplitude',
                    condition='endpoint wheel-rim speed > 0.75 cm/s; 4-cm wheel radius',
                    data_root=str(ROOT), input_arrays=str(arrays_path), input_sha256=sha256(arrays_path),
                    model_config_sha256=sha256(ROOT / 'd192/k12_seed0/config.json'),
                    split='chronological 80/10/10 of the 18-hour recording',
                    boundary_bins=bounds, bin_seconds=.02, neural_window_bins=20,
                    boundary_rule='Each input window lies entirely within its partition; no shared source bins.',
                    support=support, c_grid=GRID, regularization='L2', class_weight=None,
                    solver='lbfgs', max_iter=3000, tol=1e-5, random_state=0,
                    selection='Highest validation AUROC; ties choose smallest C; no train+validation refit.',
                    preprocessing='StandardScaler fitted on training latent amplitudes only.',
                    scope='Exploratory/transductive: frozen neural model, normalization, latent and feature '
                          'were previously fitted or selected using the full recording.',
                    sklearn_version=sklearn.__version__, numpy_version=np.__version__)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2, allow_nan=False))
    print(json.dumps(support), flush=True)
    scaler = StandardScaler().fit(values[masks['train'], None])
    xs = {name: scaler.transform(values[mask, None]) for name, mask in masks.items()}
    ys = {name: labels[mask] for name, mask in masks.items()}
    candidates = []
    fitted = []
    with threadpool_limits(limits=1):
        for c in GRID:
            started = time.monotonic()
            model = LogisticRegression(C=c, penalty='l2', solver='lbfgs',
                                       max_iter=3000, tol=1e-5, random_state=0)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(xs['train'], ys['train'])
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            probabilities = model.predict_proba(xs['validation'])[:, 1]
            auc = float(roc_auc_score(ys['validation'], probabilities))
            candidate = dict(c=c, validation_auroc=auc, converged=converged,
                             iterations=int(model.n_iter_[0]), seconds=time.monotonic()-started)
            candidates.append(candidate)
            fitted.append((auc, c, model, converged))
            print(json.dumps(candidate), flush=True)
    auc, c, model, converged = max(fitted, key=lambda row: (row[0], -row[1]))
    assert converged, 'Selected model did not converge.'
    (output / 'validation_selection.json').write_text(json.dumps(dict(chosen_c=c, candidates=candidates), indent=2))
    joblib.dump(dict(scaler=scaler, model=model), output / 'decoder.joblib')

    test_probabilities = model.predict_proba(xs['test'])[:, 1]
    test_logits = model.decision_function(xs['test'])
    test_auc = float(roc_auc_score(ys['test'], test_probabilities))
    assert np.isfinite(test_probabilities).all()
    # A one-dimensional logistic transform preserves ranking (or reverses it for a negative coefficient).
    orientation = 1 if model.coef_[0, 0] > 0 else -1
    raw_auc = float(roc_auc_score(ys['test'], values[masks['test']] * orientation))
    np.testing.assert_allclose(test_auc, raw_auc, atol=1e-12, rtol=0)
    np.testing.assert_allclose(test_auc, roc_auc_score(ys['test'], test_logits), atol=1e-12, rtol=0)
    reloaded = joblib.load(output / 'decoder.joblib')
    np.testing.assert_array_equal(test_probabilities,
        reloaded['model'].predict_proba(reloaded['scaler'].transform(values[masks['test'], None]))[:, 1])
    np.savez_compressed(output / 'test_predictions.npz', endpoint_bins=endpoints[masks['test']],
                        labels=ys['test'], probabilities=test_probabilities)
    result = dict(test_auroc=test_auc, validation_auroc=auc, chosen_c=c,
                  test_log_loss=float(log_loss(ys['test'], test_probabilities)),
                  coefficient=float(model.coef_[0, 0]), intercept=float(model.intercept_[0]),
                  scaler_mean=float(scaler.mean_[0]), scaler_scale=float(scaler.scale_[0]),
                  support=support, full_recording_direct_latent_auroc=float(roc_auc_score(labels[valid], values[valid])),
                  checks=['Original wheel labels and validity independently verified against behavior.',
                          'No shared input bins across partitions.', 'Every fitted candidate converged.' if all(x['converged'] for x in candidates) else 'See candidate convergence statuses.',
                          'Test AUROC equals monotonic scalar ranking AUROC.', 'Saved decoder reload reproduces predictions exactly.'],
                  scope=protocol['scope'])
    (output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('RESULT', json.dumps(result), flush=True)
    summary = (f'# Aeon Figure 5 single-latent decoder\n\n'
               f'Latent 52 (D=192, k=12), wheel-rim speed >0.75 cm/s.\n\n'
               f'Test AUROC: **{test_auc:.9f}**. Validation AUROC: {auc:.9f}. Selected C: {c:g}.\n\n'
               'Chronological train/validation/test intervals are the first 14.4 hours, next 1.8 hours, '
               'and final 1.8 hours. Input windows crossing these boundaries are excluded. '
               'Standardization and logistic coefficients are fitted on training data only; '
               'regularization is selected on validation AUROC without refitting.\n\n'
               f'Test support: {support["test"]["windows"]:,} windows, including '
               f'{support["test"]["positives"]:,} positives.\n\n{protocol["scope"]}\n')
    (output / 'README.md').write_text(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
