"""Training inputs never see held-out trials or blocks (uses the frozen data when present)."""

import numpy as np
import pytest

from experiments.rebuttal.change06_heldout import aeon_split, macaque_run

pytestmark = pytest.mark.skipif(not macaque_run.DATA.exists(), reason="frozen macaque data not available")


def test_macaque_training_windows_and_normalization_are_training_only():
    train, everything, anchors, scale, mean = macaque_run.build_datasets()
    with np.load(macaque_run.DATA / "metadata.npz") as meta:
        split, trials = meta["split"], meta["trial_id"]
    counts = np.load(macaque_run.DATA / "counts.npy").astype(np.float32)
    train_anchors = np.asarray(train.valid_starts, dtype=np.int64) + 9
    assert np.all(split[train_anchors] == 0)
    assert 0.79 < len(train) / len(everything) < 0.81
    # every row of every training window belongs to one training trial
    rows = train_anchors[:, None] + np.arange(-9, 1)
    assert np.all(split[rows] == 0) and np.all(trials[rows] == trials[train_anchors][:, None])
    # statistics equal those of training rows alone and differ from the all-row statistics
    training = split == 0
    np.testing.assert_allclose(mean[0], counts[training].mean(0), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(scale[0], np.maximum(counts[training].std(0), 1e-8), rtol=1e-4)
    assert not np.allclose(mean[0], counts.mean(0), rtol=1e-6, atol=0)
    # held-out trial split is whole trials: no trial is split across parts
    for t in np.unique(trials)[:200]:
        assert len(np.unique(split[trials == t])) == 1


def test_aeon_split_has_whole_blocks_and_expected_fraction():
    held = aeon_split.heldout_bins()
    blocks = held.reshape(-1, aeon_split.BLOCK_BINS)
    assert np.all(blocks.all(1) | ~blocks.any(1))  # each block is entirely held-out or training
    assert blocks.all(1).sum() == 14 and 0.19 < held.mean() < 0.2
