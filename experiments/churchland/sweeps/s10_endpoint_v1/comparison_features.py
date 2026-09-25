"""Shared development-only labels and rate-matched dense feature scoring.

All four exploratory labels are frozen before baseline selection and test decoding.
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from beartype import beartype
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[4]
SWEEP = Path('/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1')
EXPECTED_FEATURES = ('recent_braking', 'early_leftward', 'fast_target_specific',
                     'late_target_specific')


@dataclass(frozen=True)
class FeatureData:
    """Aligned evaluation rows; no test labels are included."""

    rows: np.ndarray
    labels: np.ndarray
    trials: np.ndarray
    reference_rate: float
    definition: dict


@beartype
def load_development_features() -> dict:
    """Load the four verified development feature definitions."""
    with np.load(SWEEP / 'shared/index.npz') as data:
        anchors = data['anchor_index']
    path = ROOT / 'experiments/churchland/data/nitschke_20090812'
    with np.load(path / 'metadata.npz') as data:
        rows = np.flatnonzero(data['split'][anchors] != 2)
        source = anchors[rows, None] + np.arange(-9, 1)
        trials = data['trial_id'][anchors[rows]]
        assert (data['trial_id'][source] == trials[:, None]).all()
        assert (data['split'][source] != 2).all()
        np.testing.assert_allclose(np.diff(data['timestamps'][source], axis=1), .05,
                                   atol=1e-7)
        speed = data['speed'][source]
        vx, vy = data['vel_x'][source], data['vel_y'][source]
        tx, ty = data['target_x'][anchors[rows]], data['target_y'][anchors[rows]]
        timestamp = data['timestamps'][anchors[rows]]
    with (path / 'trials.csv').open() as stream:
        table = {int(t['trial_id']): t for t in csv.DictReader(stream)
                 if int(t['split']) != 2}
    onset = np.array([float(table[int(t)]['movement_onset']) for t in trials])
    end = np.array([float(table[int(t)]['movement_end']) for t in trials])
    finite = np.isfinite(speed).all(1) & np.isfinite(vx).all(1) & np.isfinite(vy).all(1)
    target_valid = (np.isfinite(tx) & np.isfinite(ty) & np.isfinite(timestamp)
                    & np.isfinite(onset) & np.isfinite(end))
    times = np.arange(5) * .05
    times -= times.mean()
    slope = speed[:, -5:] @ times / (times @ times)
    relative = timestamp - onset
    definitions = [
        ('recent_braking', 31, finite,
         slope <= -77.71165322032654,
         'OLS speed slope over last 5 bins <= -77.71165322032654 native units/s^2'),
        ('early_leftward', 132, finite & target_valid,
         (relative >= .075) & (relative < .150) & (vx[:, -1] < 0),
         '75 <= ms since movement onset < 150 and endpoint vel_x < 0'),
        ('fast_target_specific', 181, finite,
         (speed[:, -1] >= 546.0239140959375) & (vx[:, -1] < 0)
         & (np.abs(vx[:, -1]) >= np.abs(vy[:, -1])) & (tx == -144) & (ty == -52),
         'endpoint speed >= 546.0239140959375 native units/s, heading within '
         '+/-45 degrees of leftward, target (-144,-52); no timing restriction'),
        ('late_target_specific', 4, target_valid,
         (tx == 125) & (ty == -18) & (relative >= .25) & (timestamp < end + .5),
         'target (125,-18), endpoint >= onset+0.25s and < movement_end+0.5s'),
    ]
    reference = np.load(SWEEP / 'nldisco/d256_k16_l2_seed0/activations_level_256.npy',
                        mmap_mode='r')
    return {name: FeatureData(rows[valid], labels[valid], trials[valid],
                             float((reference[rows[valid], latent] > 0).mean()),
                             dict(name=name, text=text, reference_latent=latent,
                                  reference_level=256, partition='development split != 2'))
            for name, latent, valid, labels, text in definitions}


@beartype
def score_dense(values: np.ndarray, feature: FeatureData) -> list:
    """Score a FULL export after selecting feature.rows; sign/threshold use dev only."""
    x = np.asarray(values[feature.rows])
    y = feature.labels.astype(bool)
    if x.ndim != 2 or not np.isfinite(x).all() or not y.any() or y.all():
        raise ValueError('Expected finite dense columns and both feature classes.')
    if not 0 < feature.reference_rate < 1:
        raise ValueError('Reference activity rate must be strictly between zero and one.')
    positive, negative = int(y.sum()), int((~y).sum())
    raw_auc = ((rankdata(x, axis=0, method='average')[y].sum(0)
                - positive * (positive + 1) / 2) / (positive * negative))
    sign = np.where(raw_auc < .5, -1, 1)
    oriented = x * sign
    threshold = np.quantile(oriented, 1 - feature.reference_rate, axis=0)
    active = oriented > threshold
    tpr, fpr = active[y].mean(0), active[~y].mean(0)
    sel = np.divide(tpr, tpr + fpr, out=np.zeros_like(tpr), where=tpr + fpr > 0)
    order = np.argsort(feature.trials[y], kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(feature.trials[y][order])) + 1]
    support = np.logical_or.reduceat(active[y][order], starts, axis=0).sum(0)
    varying = np.ptp(x, axis=0) > 0
    return [dict(latent_id=i, orientation=int(sign[i]), threshold=float(threshold[i]),
                 reference_rate=feature.reference_rate, achieved_rate=float(active[:, i].mean()),
                 sel=float(sel[i]), auroc=float(max(raw_auc[i], 1 - raw_auc[i])),
                 tpr=float(tpr[i]), fpr=float(fpr[i]),
                 active_positive_trials=int(support[i]), positive_bins=positive,
                 negative_bins=negative, varying=bool(varying[i]),
                 eligible=bool(varying[i] and tpr[i] >= .05 and support[i] >= 20))
            for i in range(x.shape[1])]


@beartype
def best_eligible(rows: list):
    """Use best AUROC, then lowest latent ID; no extra selectivity cutoff."""
    return max((r for r in rows if r['eligible']),
               key=lambda r: (r['auroc'], -r['latent_id']), default=None)


@beartype
def configuration_score(winners: dict) -> float:
    """Reject incomplete scores rather than rank configurations on only three labels."""
    if set(winners) != set(EXPECTED_FEATURES):
        raise ValueError('All four verified feature definitions are required.')
    if any(winner is None for winner in winners.values()):
        raise ValueError('Configuration has a feature without an eligible latent.')
    return float(np.mean([winners[name]['auroc'] for name in EXPECTED_FEATURES]))
