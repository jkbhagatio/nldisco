"""Fast checks of window sampling, normalization, and the strict endpoint model."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

MODULE = Path(__file__).with_name("shared_training.py")
SPEC = importlib.util.spec_from_file_location("aeon_shared_training", MODULE)
training = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = training
SPEC.loader.exec_module(training)


def test_normalization_uses_only_valid_bins():
    counts = np.array([[1, 2], [3, 2], [99, 100]], dtype=np.uint16)
    mean, std, n = training.compute_statistics(counts, np.array([True, True, False]))
    np.testing.assert_array_equal(mean, [2, 2])
    np.testing.assert_array_equal(std, [1, 1])
    assert n == 2


def test_lazy_sampling_full_coverage_and_calibration():
    counts = np.arange(100, dtype=np.uint16).reshape(50, 2)
    starts = np.array([0, 1, 2, 10, 11, 12, 20])
    loader = training.WindowBatches(counts, starts, np.zeros(2), np.ones(2), 4, 3, 0)
    batches = list(loader)
    assert len(batches) == 3
    assert loader.seen.sum() == 12
    assert loader.seen.min() == 1 and loader.seen.max() == 2
    for batch in batches:
        np.testing.assert_array_equal(batch.values.numpy(), counts[batch.source_indices.numpy()])
        assert batch.occurrence_mask.sum() == len(batch.values)
        assert batch.occurrence_mask[:, -1].all()
    calibration = list(loader)
    indices = torch.cat([batch.window_index for batch in calibration]).numpy()
    np.testing.assert_array_equal(np.sort(indices), np.arange(len(starts)))


def test_endpoint_code_support_and_reload(tmp_path):
    torch.set_num_threads(1)
    cfg = training.make_config(16, 4, 2)
    assert cfg.temporal_occurrence_support() == (19, 0)
    model = training.build_sed(cfg)
    x = torch.randn(8, 20, 2)
    result = model(x)
    assert not result.acts[:, :-1].any()
    assert result.sparse_acts[16].shape == (8, 20, 16)
    assert result.reconstructions[16].shape == x.shape
    model.sparsifier.begin_threshold_calibration()
    model.eval()
    model(x)
    model.sparsifier.end_threshold_calibration()
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    restored = training.build_sed(cfg).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(model(x).sparse_acts[16], restored(x).sparse_acts[16])


def test_training_export_roundtrip(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    np.save(data / "counts.npy", np.random.default_rng(0).poisson(2, (50, 2)).astype(np.uint16))
    np.save(data / "valid_starts.npy", np.arange(31))
    np.save(data / "neural_valid.npy", np.ones(50, dtype=bool))
    (data / "READY.json").write_text("{}")
    output = tmp_path / "run"
    training.run(argparse.Namespace(data=str(data), output=str(output), d=16, k=4,
                                    steps=3, batch_size=16, seed=0,
                                    gpu_memory_gb=8., device="cpu"))
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["checkpoint_reload_exact"]
    assert metrics["sampler_exposure"]["coverage"] == 1.
    np.testing.assert_array_equal(np.load(output / "endpoint_bins.npy"), np.arange(19, 50))
    sparse = np.load(output / "activations.npy")
    dense = np.load(output / "prethreshold_activations.npy")
    np.testing.assert_array_equal(sparse, np.where(dense >= metrics["thresholds"]["16"], dense, 0.))
    assert (output / "COMPLETE.json").is_file()


def test_validation_window_does_not_advance_sampler():
    counts = np.arange(100, dtype=np.uint16).reshape(50, 2)
    loader = training.WindowBatches(counts, np.arange(31), np.zeros(2), np.ones(2), 4, 3, 0)
    sample = loader.dataset[0]
    assert len(loader.dataset) == 31
    np.testing.assert_array_equal(sample.values, counts[:20])
    assert sample.target is sample.values or torch.equal(sample.target, sample.values)
    assert loader.iterations == 0 and loader.seen.sum() == 0
