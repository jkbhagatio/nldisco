"""Temporal split checks for the Aeon appendix's behavioral decoders."""

import joblib
import numpy as np
import pytest

from experiments.aeon import fit_appendix_decoders as decoders
from experiments.aeon.fit_appendix_decoders import split_masks


def test_splits_exclude_shared_source_bins_and_preserve_physical_boundaries():
    endpoints = np.arange(2, 30, dtype=np.int64)
    valid = np.ones(len(endpoints), dtype=bool)
    valid[endpoints < 8] = False
    masks = split_masks(endpoints, valid, bounds=(0, 20, 25, 30), width=3)
    np.testing.assert_array_equal(endpoints[masks["train"]], np.arange(8, 20))
    np.testing.assert_array_equal(endpoints[masks["validation"]], np.arange(22, 25))
    np.testing.assert_array_equal(endpoints[masks["test"]], np.arange(27, 30))
    source_bins = [set((endpoints[mask, None] - np.arange(3)).ravel())
                   for mask in masks.values()]
    assert all(not source_bins[i] & source_bins[j] for i, j in ((0, 1), (0, 2), (1, 2)))


def test_split_rejects_unsorted_endpoints():
    with pytest.raises(ValueError, match="strictly increasing"):
        split_masks(np.array([2, 4, 3]), np.ones(3, dtype=bool))


def test_fits_ignore_test_values_and_fit_scaler_on_training_only(tmp_path, monkeypatch):
    rng = np.random.default_rng(75)
    endpoints = np.arange(2, 160, dtype=np.int64)
    values = rng.exponential(size=(len(endpoints), 4)).astype(np.float32)
    labels = values[:, 1] + .5 * values[:, 2] > 1.5
    valid = np.ones(len(endpoints), dtype=bool)
    masks = split_masks(endpoints, valid, bounds=(0, 100, 130, 160), width=3)
    run = tmp_path / "d4/k12_seed0"
    run.mkdir(parents=True)
    np.save(run / "endpoint_bins.npy", endpoints)
    monkeypatch.setattr(decoders, "ROOT", tmp_path)
    monkeypatch.setattr(decoders, "SOURCE", tmp_path)
    monkeypatch.setitem(decoders.FEATURES, "area", (4, 1))
    monkeypatch.setattr(decoders, "split_masks", lambda endpoints, valid: masks)
    results = []
    for iteration in range(2):
        if iteration:
            values[masks["test"]] += 1e6
            labels[masks["test"]] = ~labels[masks["test"]]
        np.save(run / "activations.npy", values)
        np.savez(tmp_path / "feature_arrays.npz", endpoint_bins=endpoints,
                 values_area=values[:, 1], condition_area=labels, valid_area=valid)
        output = tmp_path / f"fit{iteration}"
        output.mkdir()
        rows = decoders.fit_feature("area", output, threads=1)
        results.append([(row, joblib.load(row["model_path"])) for row in rows])
    for (before, fit_before), (after, fit_after) in zip(*results):
        assert before["chosen_c"] == after["chosen_c"]
        assert before["threshold"] == after["threshold"]
        np.testing.assert_array_equal(fit_before["model"].coef_, fit_after["model"].coef_)
        columns = [1] if before["space"] == "single" else list(range(4))
        np.testing.assert_allclose(fit_after["scaler"].mean_,
                                   values[masks["train"]][:, columns].astype(float).mean(axis=0))
