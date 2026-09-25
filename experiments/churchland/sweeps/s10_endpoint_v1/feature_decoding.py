"""Frozen S10 single-coordinate and full-space logistic decoding.

Stages: --freeze binds data, sources, feature definitions and exact baseline
illustrative columns; --evaluate fits train/validation probes before opening test
outcomes once. --report only reads the saved test results.
"""
from __future__ import annotations

import csv
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from beartype import beartype
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
SWEEP = Path('/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1')
DATA = ROOT / 'experiments/churchland/data/nitschke_20090812'
FEATURES = ('recent_braking', 'early_leftward', 'fast_target_specific', 'late_target_specific')
LATENTS = dict(zip(FEATURES, (31, 132, 181, 4)))
LABEL_TEXT = {
    'recent_braking': 'OLS speed slope over last 5 bins <= -77.71165322032654 native units/s^2',
    'early_leftward': '75 <= ms since movement onset < 150 and endpoint vel_x < 0',
    'fast_target_specific': 'endpoint speed >= 546.0239140959375 native units/s, heading within +/-45 degrees of leftward, target (-144,-52); no timing restriction',
    'late_target_specific': 'target (125,-18), endpoint >= onset+0.25s and < movement_end+0.5s',
}
NAMES = {'nldisco': 'NLDisco', 'cebra': 'CEBRA-Time', 'langevinflow': 'LangevinFlow'}
C_GRID = (.001, .01, .1, 1., 10., 100., 1000.)


@beartype
def sha256(path: Path) -> str:
    """Hash an artifact incrementally."""
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


@beartype
def write_json(path: Path, value: dict, exclusive: bool = False) -> None:
    """Write an explicit machine-readable record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


@beartype
def label_windows(speed: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                  tx: np.ndarray, ty: np.ndarray, timestamp: np.ndarray,
                  onset: np.ndarray, end: np.ndarray) -> dict:
    """Apply the four fixed definitions with exactly the discovery validity masks."""
    finite = np.isfinite(speed).all(1) & np.isfinite(vx).all(1) & np.isfinite(vy).all(1)
    target_valid = (np.isfinite(tx) & np.isfinite(ty) & np.isfinite(timestamp)
                    & np.isfinite(onset) & np.isfinite(end))
    times = np.arange(5) * .05
    times -= times.mean()
    slope = speed[:, -5:] @ times / (times @ times)
    relative = timestamp - onset
    return {
        'recent_braking': (finite, slope <= -77.71165322032654),
        'early_leftward': (finite & target_valid,
                           (relative >= .075) & (relative < .150) & (vx[:, -1] < 0)),
        'fast_target_specific': (finite, (speed[:, -1] >= 546.0239140959375)
                                  & (vx[:, -1] < 0) & (np.abs(vx[:, -1]) >= np.abs(vy[:, -1]))
                                  & (tx == -144) & (ty == -52)),
        'late_target_specific': (target_valid, (tx == 125) & (ty == -18)
                                  & (relative >= .25) & (timestamp < end + .5)),
    }


@beartype
def load_labels(sweep: Path, data_path: Path, partitions: tuple) -> dict:
    """Load only requested label partitions while preserving full export row IDs."""
    with np.load(sweep / 'shared/index.npz') as archive:
        anchors = archive['anchor_index']
        assert len(anchors) == 130507
        source_all = archive['source_indices']
    with np.load(data_path / 'metadata.npz') as archive:
        rows = np.flatnonzero(np.isin(archive['split'][anchors], partitions))
        source = anchors[rows, None] + np.arange(-9, 1)
        np.testing.assert_array_equal(source, source_all[rows])
        trials = archive['trial_id'][anchors[rows]]
        split = archive['split'][anchors[rows]]
        assert (archive['trial_id'][source] == trials[:, None]).all()
        assert (archive['split'][source] == split[:, None]).all()
        np.testing.assert_allclose(np.diff(archive['timestamps'][source], axis=1), .05, atol=1e-7)
        speed, vx, vy = (archive[key][source] for key in ('speed', 'vel_x', 'vel_y'))
        tx, ty, timestamp = (archive[key][anchors[rows]] for key in ('target_x', 'target_y', 'timestamps'))
    with (data_path / 'trials.csv').open() as stream:
        table = {int(row['trial_id']): row for row in csv.DictReader(stream)
                 if int(row['split']) in partitions}
    onset = np.array([float(table[int(trial)]['movement_onset']) for trial in trials])
    end = np.array([float(table[int(trial)]['movement_end']) for trial in trials])
    definitions = label_windows(speed, vx, vy, tx, ty, timestamp, onset, end)
    return {name: dict(rows=rows[valid], labels=labels[valid], trials=trials[valid], split=split[valid])
            for name, (valid, labels) in definitions.items()}




@beartype
def load_representation(record: dict, sweep: Path) -> np.ndarray:
    """Verify ordered endpoint identity and finite activations."""
    path = Path(record['path'])
    if path.suffix == '.npz':
        with np.load(path) as archive, np.load(sweep / 'shared/index.npz') as index:
            np.testing.assert_array_equal(archive['source_index'], index['anchor_index'])
            values = archive['activations']
    else:
        values = np.load(path, mmap_mode='r')
        index_path = path.parent / ('activation_index.npz' if 'activations_level_' in path.name else 'index.npz')
        with np.load(index_path) as archive, np.load(sweep / 'shared/index.npz') as index:
            np.testing.assert_array_equal(archive['source_index' if 'source_index' in archive else 'anchor_index'],
                                          index['anchor_index'])
    if values.ndim != 2 or len(values) != 130507 or not np.isfinite(values).all():
        raise ValueError(f'Invalid representation: {path}')
    return values




@beartype
def tune_probe(values: np.ndarray, feature: dict, c_grid: tuple = C_GRID) -> tuple:
    """Tune without reading test labels; the supplied feature has train/val only."""
    if not np.isin(feature['split'], (0, 1)).all():
        raise ValueError('Probe tuning accepts development rows only.')
    x = np.asarray(values[feature['rows']], dtype=np.float64)
    y = feature['labels']
    train, validation = feature['split'] == 0, feature['split'] == 1
    if len(np.unique(y[train])) != 2 or len(np.unique(y[validation])) != 2:
        raise ValueError('Both classes are required in each development split.')
    scaler = StandardScaler().fit(x[train])
    x_train, x_validation = scaler.transform(x[train]), scaler.transform(x[validation])
    fitted = []
    for c in c_grid:
        model = LogisticRegression(C=c, penalty='l2', solver='lbfgs', fit_intercept=True,
                                   class_weight=None, max_iter=3000, tol=1e-5, random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x_train, y[train])
        converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
        score = float(roc_auc_score(y[validation], model.predict_proba(x_validation)[:, 1]))
        fitted.append((score, float(c), model, converged))
    score, c, model, converged = max(fitted, key=lambda item: (item[0], -item[1]))
    if not converged:
        raise RuntimeError(f'Selected validation probe C={c} did not converge; test remains closed.')
    counts = {f'{label}_{key}': int(value) for label, mask in (('train', train), ('validation', validation))
              for key, value in (('rows', mask.sum()), ('positives', y[mask].sum()),
                                 ('trials', len(np.unique(feature['trials'][mask]))))}
    return (scaler, model), dict(chosen_c=c, validation_auroc=score, **counts,
                                grid=[dict(c=candidate[1], validation_auroc=candidate[0], converged=candidate[3])
                                      for candidate in fitted])


@beartype
def trial_interval(labels: np.ndarray, predictions: np.ndarray, trials: np.ndarray,
                   repetitions: int, seed: int) -> tuple:
    """Resample entire test trials, not overlapping endpoint windows independently."""
    blocks = [np.flatnonzero(trials == trial) for trial in np.unique(trials)]
    rng, scores = np.random.default_rng(seed), []
    for _ in range(repetitions):
        rows = np.concatenate([blocks[index] for index in rng.integers(len(blocks), size=len(blocks))])
        if len(np.unique(labels[rows])) == 2:
            scores.append(roc_auc_score(labels[rows], predictions[rows]))
    low, high = np.quantile(scores, [.025, .975]) if scores else (np.nan, np.nan)
    return float(low), float(high), len(scores)








