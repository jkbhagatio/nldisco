"""Synthetic tests for sorter ingestion, exact binning, and normalization."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal

from nldisco.data import (
    SpikeWindowDataset,
    bin_spikes,
    fit_normalizer,
    load_kilosort,
    normalize_activity,
)


def write_sorting(path, samples=(0, 9, 10, 19, 20, 24, 25), column=False):
    times = np.array(samples, dtype=np.int64)
    units = np.array([9, 2, 9, 2, 9, 9, 2], dtype=np.int32)
    np.save(path / "spike_times.npy", times[:, None] if column else times)
    np.save(path / "spike_clusters.npy", units[:, None] if column else units)


@pytest.mark.parametrize("column", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3, 100])
def test_kilosort_boundaries_partial_bin_and_cluster_mapping(tmp_path, column, chunk_size):
    write_sorting(tmp_path, column=column)
    result = load_kilosort(
        tmp_path,
        sampling_rate=100.0,
        bin_size=0.1,
        stop_time=0.25,
        chunk_size=chunk_size,
    )
    assert_array_equal(result.unit_ids, [2, 9])
    assert_array_equal(result.counts, [[1, 1], [1, 1], [0, 2]])
    assert result.counts.sum() == 6  # The spike exactly at stop_time is excluded.
    assert_allclose(result.bin_edges, [0, 0.1, 0.2, 0.25])
    assert_allclose(result.timestamps, [0.05, 0.15, 0.225])


def test_unsorted_samples_repeats_silence_and_nonzero_start(tmp_path):
    samples = np.array([11, 10, 10, 9, 30, 21])
    units = np.array([9, 9, 9, 2, 9, 2])
    result = bin_spikes(
        samples,
        units,
        sampling_rate=100.0,
        bin_size=0.1,
        start_time=0.1,
        stop_time=0.4,
        unit_ids=[12, 9, 2],
        chunk_size=2,
        output_path=tmp_path / "counts.npy",
    )
    assert_array_equal(result.unit_ids, [2, 9, 12])
    assert_array_equal(result.counts, [[0, 3, 0], [1, 0, 0], [0, 1, 0]])
    assert_array_equal(np.load(tmp_path / "counts.npy"), result.counts)
    assert isinstance(result.counts, np.memmap)


def test_empty_recording_and_empty_selection():
    for samples, units in [
        (np.array([], dtype=int), np.array([], dtype=int)),
        (np.array([40]), np.array([9])),
    ]:
        result = bin_spikes(
            samples,
            units,
            sampling_rate=100.0,
            bin_size=0.1,
            stop_time=0.3,
            unit_ids=[9],
        )
        assert_array_equal(result.counts, np.zeros((3, 1)))


def test_quality_filter_prefers_curated_labels(tmp_path):
    write_sorting(tmp_path)
    (tmp_path / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n2\tgood\n9\tmua\n")
    args = dict(sampling_rate=100.0, bin_size=0.1, stop_time=0.3, labels=["good"])
    assert_array_equal(load_kilosort(tmp_path, **args).unit_ids, [2])
    (tmp_path / "cluster_group.tsv").write_text("cluster_id\tgroup\n9\tgood\n2\tnoise\n")
    curated = load_kilosort(tmp_path, **args)
    assert_array_equal(curated.unit_ids, [9])
    assert curated.counts.sum() == 4
    intersected = load_kilosort(tmp_path, unit_ids=[2], **args)
    assert intersected.counts.shape == (3, 0)
    # No implicit filtering when neither labels nor IDs are requested.
    args.pop("labels")
    assert_array_equal(load_kilosort(tmp_path, **args).unit_ids, [2, 9])


def test_requested_labels_require_metadata(tmp_path):
    write_sorting(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_kilosort(tmp_path, sampling_rate=100.0, bin_size=0.1, stop_time=1.0, labels=["good"])


@pytest.mark.parametrize(
    "contents", ["cluster_id\tgroup\n2\tgood\n2\tmua\n", "id\tgroup\n2\tgood\n"]
)
def test_invalid_quality_metadata(tmp_path, contents):
    write_sorting(tmp_path)
    (tmp_path / "cluster_group.tsv").write_text(contents)
    with pytest.raises(ValueError):
        load_kilosort(tmp_path, sampling_rate=100.0, bin_size=0.1, stop_time=1.0, labels=["good"])


@pytest.mark.parametrize(
    "samples",
    [np.ones((2, 2), dtype=int), np.array([1.5, 2.0]), np.array([-1, 2]), np.array([1, 2, 3])],
)
def test_invalid_sorter_arrays(tmp_path, samples):
    np.save(tmp_path / "spike_times.npy", samples)
    np.save(tmp_path / "spike_clusters.npy", np.array([2, 9]))
    with pytest.raises(ValueError):
        load_kilosort(tmp_path, sampling_rate=100.0, bin_size=0.1, stop_time=1.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sampling_rate": 0.0},
        {"sampling_rate": float("nan")},
        {"bin_size": 0.0},
        {"bin_size": 0.015},
        {"start_time": -1.0},
        {"stop_time": 0.0},
        {"stop_time": float("inf")},
        {"chunk_size": 0},
        {"unit_ids": [2, 2]},
    ],
)
def test_invalid_binning_arguments(overrides):
    kwargs = dict(sampling_rate=100.0, bin_size=0.1, stop_time=0.3)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        bin_spikes(np.array([0]), np.array([2]), **kwargs)


def test_integer_bin_arithmetic_avoids_decimal_boundary_errors():
    samples = np.arange(0, 30_000, 600, dtype=np.int64)
    result = bin_spikes(
        samples, np.zeros_like(samples), sampling_rate=30_000.0, bin_size=0.02, stop_time=1.0
    )
    assert_array_equal(result.counts, np.ones((50, 1)))


def test_counts_do_not_overflow_uint16():
    result = bin_spikes(
        np.zeros(70_000, dtype=int),
        np.full(70_000, 4),
        sampling_rate=100.0,
        bin_size=0.1,
        stop_time=0.1,
    )
    assert result.counts[0, 0] == 70_000


def test_large_integer_cluster_ids_are_preserved():
    largest = np.iinfo(np.int64).max
    result = bin_spikes(
        np.array([0]),
        np.array([largest]),
        sampling_rate=100,
        bin_size=0.1,
        stop_time=0.1,
        unit_ids=np.array([largest]),
    )
    assert_array_equal(result.unit_ids, [largest])
    assert result.counts[0, 0] == 1
    with pytest.raises(ValueError, match="int64"):
        bin_spikes(
            np.array([0]),
            np.array([2**63], dtype=np.uint64),
            sampling_rate=100,
            bin_size=0.1,
            stop_time=0.1,
        )


@pytest.mark.parametrize("method", ["zscore", "minmax"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("chunk_size", [1, 3, 100])
def test_normalization_matches_independent_numpy_reference(method, axis, chunk_size):
    values = np.array([[0, 2, 4], [1, 1, 1], [2, 4, 8], [3, 2, 0]], dtype=np.int16)
    before = values.copy()
    if method == "zscore":
        offset = values.mean(axis=axis, keepdims=True)
        scale = values.std(axis=axis, keepdims=True)
    else:
        offset = values.min(axis=axis, keepdims=True)
        scale = values.max(axis=axis, keepdims=True) - offset
    expected = (values - offset) / np.where(scale == 0, 1, scale)
    result = normalize_activity(values, method, axis=axis, chunk_size=chunk_size)
    assert_allclose(result, expected, atol=1e-14)
    assert_array_equal(values, before)
    assert result.shape == values.shape
    assert result.dtype == np.float64


@pytest.mark.parametrize("method", ["zscore", "minmax"])
def test_training_statistics_are_reused_without_leakage(method):
    train = np.array([[0, 5, 0], [2, 5, 0]])
    test = np.array([[4, 7, 1]])
    fitted = fit_normalizer(train, method, chunk_size=1)
    expected = [[3, 2, 1]] if method == "zscore" else [[2, 2, 1]]
    assert_allclose(fitted.transform(test), expected)
    assert_array_equal(fitted.transform(train)[:, 1:], np.zeros((2, 2)))
    with pytest.raises(ValueError, match="unit columns"):
        fitted.transform(np.ones((2, 4)))


def test_large_offset_zscore_is_numerically_stable():
    data = 1e12 + np.arange(30, dtype=float).reshape(10, 3)
    fitted = fit_normalizer(data, chunk_size=2)
    assert_allclose(fitted.offset, data.mean(axis=0), rtol=0, atol=1e-5)
    assert_allclose(fitted.scale, data.std(axis=0), rtol=1e-12)


@pytest.mark.parametrize("method", ["zscore", "minmax"])
@pytest.mark.parametrize("axis", [0, 1])
def test_constant_activity_memmap_and_windows(tmp_path, method, axis):
    source = np.full((12, 3), 7)
    path = tmp_path / "normalized.npy"
    result = normalize_activity(source, method, axis=axis, output_path=path, chunk_size=3)
    assert isinstance(result, np.memmap)
    assert_array_equal(np.load(path), np.zeros((12, 3)))
    windows = SpikeWindowDataset(torch.tensor(result, dtype=torch.float32), seq_len=4)
    assert windows[0].values.shape == (4, 3)
    assert len(windows) == 9
    with pytest.raises(FileExistsError):
        normalize_activity(source, method, axis=axis, output_path=path)
    assert_array_equal(np.load(path), np.zeros((12, 3)))


@pytest.mark.parametrize(
    "values", [np.array([[np.nan]]), np.array([[np.inf]]), np.empty((0, 2)), np.empty((2, 0))]
)
@pytest.mark.parametrize("method", ["zscore", "minmax"])
@pytest.mark.parametrize("axis", [0, 1])
def test_invalid_normalization_activity(values, method, axis):
    with pytest.raises(ValueError):
        normalize_activity(values, method, axis=axis)


def test_invalid_normalization_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        fit_normalizer(np.ones((2, 3)), chunk_size=0)


def test_binning_never_overwrites_an_existing_output(tmp_path):
    output = tmp_path / "source.npy"
    np.save(output, [17])
    with pytest.raises(FileExistsError):
        bin_spikes(
            np.array([0]),
            np.array([2]),
            sampling_rate=100.0,
            bin_size=0.1,
            stop_time=0.3,
            output_path=output,
        )
    assert_array_equal(np.load(output), [17])


def test_kilosort_label_header_in_cluster_group_and_numpy_id_selection(tmp_path):
    write_sorting(tmp_path)
    (tmp_path / "cluster_group.tsv").write_text("cluster_id\tKSLabel\n2\tgood\n9\tmua\n")
    args = dict(sampling_rate=100, bin_size=0.1, stop_time=1, unit_ids=np.array([2, 9]))
    selected = load_kilosort(tmp_path, labels=["good"], **args)
    assert_array_equal(selected.unit_ids, [2])
    assert selected.counts.sum() == 3
    empty = load_kilosort(tmp_path, labels=["noise"], output_path=tmp_path / "empty.npy", **args)
    assert empty.counts.shape == (10, 0)
