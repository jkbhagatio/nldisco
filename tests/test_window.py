import pandas as pd
import pytest
import torch as t

from nldisco.data.window import SpikeWindowDataset


def test_windows_do_not_cross_trial_boundaries():
    counts = t.arange(24).reshape(8, 3)
    dataset = SpikeWindowDataset(
        counts,
        seq_len=3,
        trial_ids=[0, 0, 0, 0, 1, 1, 1, 1],
    )
    assert dataset.valid_starts == [0, 1, 4, 5]
    assert dataset[2].anchor_index.item() == 6


def test_windows_do_not_cross_session_or_timestamp_gaps():
    counts = t.zeros(7, 2)
    dataset = SpikeWindowDataset(
        counts,
        seq_len=2,
        session_ids=[0, 0, 0, 0, 1, 1, 1],
        timestamps=[0.0, 0.1, 0.2, 0.5, 1.0, 1.1, 1.2],
        expected_bin_size=0.1,
    )
    assert dataset.valid_starts == [0, 1, 4, 5]


def test_timestamps_require_one_fixed_expected_bin_size():
    counts = t.zeros(4, 2)
    with pytest.raises(ValueError, match="expected_bin_size"):
        SpikeWindowDataset(counts, seq_len=2, timestamps=[0, 100, 101, 500])
    dataset = SpikeWindowDataset(
        counts,
        seq_len=2,
        timestamps=[0, 100, 101, 500],
        expected_bin_size=1,
    )
    assert dataset.valid_starts == [1]


def test_missing_group_ids_are_rejected():
    counts = t.zeros(3, 2)
    dataset = SpikeWindowDataset(counts, seq_len=2, trial_ids=[0, pd.NA, pd.NA])
    assert dataset.valid_starts == []


def test_allowed_trials_are_applied_before_windowing():
    counts = t.zeros(6, 2)
    dataset = SpikeWindowDataset(
        counts,
        seq_len=2,
        trial_ids=["train", "train", "train", "val", "val", "val"],
        allowed_trial_ids={"val"},
    )
    assert dataset.valid_starts == [3, 4]


def test_stride_restarts_at_each_contiguous_group():
    counts = t.arange(14).reshape(7, 2)
    dataset = SpikeWindowDataset(
        counts,
        seq_len=3,
        stride=3,
        trial_ids=[0, 1, 1, 1, 2, 2, 2],
    )
    assert dataset.valid_starts == [1, 4]
    assert t.equal(dataset[0].values, counts[1:4])
    assert t.equal(dataset[1].values, counts[4:7])


def test_dataset_rejects_nonfinite_expected_bin_size():
    with pytest.raises(ValueError, match="finite and positive"):
        SpikeWindowDataset(
            t.zeros(3, 2),
            seq_len=2,
            timestamps=[0.0, 0.1, 0.2],
            expected_bin_size=float("nan"),
        )


def test_occurrence_support_covers_group_tails_without_duplicates():
    counts = t.arange(36).reshape(18, 2)
    dataset = SpikeWindowDataset(
        counts,
        seq_len=5,
        stride=3,
        trial_ids=[0] * 9 + [1] * 9,
        occurrence_support=(2, 0),
    )
    assert dataset.valid_starts == [0, 3, 4, 9, 12, 13]
    eligible_sources = []
    for sample in dataset:
        eligible_sources.extend(sample.source_indices[sample.occurrence_mask].tolist())
    assert eligible_sources == list(range(2, 9)) + list(range(11, 18))
    assert len(eligible_sources) == len(set(eligible_sources))
