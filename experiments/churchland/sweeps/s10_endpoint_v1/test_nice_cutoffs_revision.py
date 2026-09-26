"""Regression checks for rounded labels and preserved selection conventions."""
import numpy as np
from sklearn.metrics import roc_auc_score

from . import nice_cutoffs_revision as revision
from .comparison_features import FeatureData


def test_revision_changes_only_intended_labels():
    old, new = revision.OLD_FEATURES(), revision.features()
    for name in old:
        for key in ('rows', 'trials'):
            np.testing.assert_array_equal(getattr(old[name], key), getattr(new[name], key))
        assert old[name].reference_rate == new[name].reference_rate
        if name in revision.CHANGED:
            assert np.all(~old[name].labels | new[name].labels)
            assert new[name].labels.sum() > old[name].labels.sum()
        else:
            np.testing.assert_array_equal(old[name].labels, new[name].labels)
    partitioned = revision.labels(revision.shared.SWEEP, revision.decoder.DATA, (0, 1))
    for name in new:
        np.testing.assert_array_equal(partitioned[name]['labels'], new[name].labels)


def test_rescore_retains_orientation_and_strict_threshold():
    values = np.array([[3.], [2.], [1.], [0.]])
    feature = FeatureData(np.arange(4), np.array([True, True, False, False]), np.arange(4), .25, {})
    previous = dict(latent_id=0, orientation=-1, threshold=-2.)
    row = revision.rescore(values, feature, previous)
    assert row['orientation'] == -1
    assert row['threshold'] == -2.
    assert row['tpr'] == 0.
    assert row['fpr'] == 1.
    assert row['auroc'] == roc_auc_score(feature.labels, -values[:, 0])
    assert row['auroc'] == 0.  # Do not silently reorient under revised labels.
