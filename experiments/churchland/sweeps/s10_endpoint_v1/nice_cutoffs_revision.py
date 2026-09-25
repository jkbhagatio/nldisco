"""Current Churchland feature labels and rescoring of frozen selections."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from . import comparison_features as shared
from . import feature_decoding as decoder
ROOT = decoder.ROOT / 'experiments/churchland/outputs'
PREVIOUS = ROOT / 's10_endpoint_revision_v3_192'
OUTPUT = ROOT / 's10_endpoint_revision_v4_nice_cutoffs'
CHANGED = ('recent_braking', 'fast_target_specific')
OLD_FEATURES = shared.load_development_features
OLD_LABELS = decoder.load_labels
TEXT = dict(recent_braking='OLS speed slope over last 5 bins <= -75 mm/s^2',
            fast_target_specific='endpoint speed >= 500 mm/s, heading within +/-45 degrees of leftward, target (-144,-52); no timing restriction')

def revised_labels(rows: np.ndarray) -> dict:
    """Compute the two new labels on already validated endpoint rows."""
    with np.load(shared.SWEEP / 'shared/index.npz') as index:
        anchors = index['anchor_index'][rows]
    with np.load(decoder.DATA / 'metadata.npz') as meta:
        times = np.arange(5) * .05
        times -= times.mean()
        slope = meta['speed'][anchors[:, None] + np.arange(-4, 1)] @ times / (times @ times)
        vx, vy = meta['vel_x'][anchors], meta['vel_y'][anchors]
        return dict(recent_braking=slope <= -75,
                    fast_target_specific=(meta['speed'][anchors] >= 500) & (vx < 0)
                    & (np.abs(vx) >= np.abs(vy)) & (meta['target_x'][anchors] == -144)
                    & (meta['target_y'][anchors] == -52))


def features() -> dict:
    """Retain historical validity masks and activity-reference rates."""
    result = OLD_FEATURES()
    for name in CHANGED:
        f = result[name]
        result[name] = replace(f, labels=revised_labels(f.rows)[name],
                               definition=dict(f.definition, text=TEXT[name]))
    return result


def labels(sweep: Path, data_path: Path, partitions: tuple) -> dict:
    """Keep the original whole-trial partitions and behavioral validity masks."""
    result = OLD_LABELS(sweep, data_path, partitions)
    for name in CHANGED:
        result[name]['labels'] = revised_labels(result[name]['rows'])[name]
    return result


def rescore(values: np.ndarray, feature: shared.FeatureData, previous: dict) -> dict:
    """Rescore an unchanged dimension/sign/activity threshold on new labels."""
    z = np.asarray(values[feature.rows, previous['latent_id']]) * previous['orientation']
    active, y = z > previous['threshold'], feature.labels
    tpr, fpr = float(active[y].mean()), float(active[~y].mean())
    support = len(np.unique(feature.trials[active & y]))
    return dict(previous, sel=tpr / (tpr + fpr) if tpr + fpr else 0.,
                auroc=float(roc_auc_score(y, z)), tpr=tpr, fpr=fpr,
                active_positive_trials=support, positive_bins=int(y.sum()),
                negative_bins=int((~y).sum()), active_positive_bins=int((active & y).sum()),
                precision=float(y[active].mean()) if active.any() else 0.,
                achieved_rate=float(active.mean()), varying=bool(np.ptp(z)),
                eligible=bool(np.ptp(z) and tpr >= .05 and support >= 20))

