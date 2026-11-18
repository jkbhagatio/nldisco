"""Fast neural-only checks for ST-NDT windows, causality, masking and export."""
import json

import pytest

pytest.importorskip("yacs", reason="Use the documented ST-NDT overlay for this test")

import numpy as np
import torch

from experiments.rebuttal.change01_stndt.environments.setup_stndt import verify_source
from experiments.rebuttal.change01_stndt.macaque import configurations
from experiments.rebuttal.change01_stndt.stndt import (
    UNMASKED_LABEL,
    Hyperparameters,
    Masker,
    extract,
    make_config,
    make_model,
    masked,
    train_and_export,
    warmup_cosine,
)

UNITS, BINS = 6, 10


def _model(hp=Hyperparameters()):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    return make_model(UNITS, BINS, hp, "cpu", 9).eval()


def test_pinned_source_and_sweep():
    verify_source()
    ids = [config["config_id"] for config in configurations()]
    assert len(ids) == len(set(ids)) == 12
    config = make_config(Hyperparameters(contrast_weight=.5, dropout=.3))
    assert config.MODEL.LAMBDA == .5 and config.MODEL.DROPOUT_RATES == .3
    assert config.MODEL.FULL_CONTEXT and config.TRAIN.MASK_RATIO == .25
    with pytest.raises(ValueError, match="must divide"):
        make_model(UNITS, BINS, Hyperparameters(heads=4), "cpu", 9)


def test_window_alignment_and_export_shape():
    model = _model()
    counts = torch.randint(0, 5, (40, UNITS))
    indices = np.arange(9, 40)[:, None] + np.arange(-9, 1)
    exported = extract(model, counts, indices, 7)
    assert exported.shape == (31, UNITS) and exported.dtype == np.float32
    spikes = counts[torch.as_tensor(indices)]
    with torch.no_grad():
        layers = model(spikes, spikes, val_phase=True, return_outputs=True,
                       return_weights=True)[5]
    assert layers.shape == (31, BINS, UNITS, 4)
    # One vector per window: the final layer at the window's last bin (row anchor t).
    np.testing.assert_allclose(exported, layers[:, -1, :, -1].numpy(), atol=1e-6)
    assert not np.allclose(exported, layers[:, -2, :, -1].numpy(), atol=1e-3)


def test_no_access_outside_window():
    model = _model()
    counts = torch.randint(0, 5, (30, UNITS))
    indices = np.array([np.arange(5, 15)])
    reference = extract(model, counts, indices, 1)
    outside = counts.clone()
    outside[:5] += 3   # bins before the window
    outside[15:] += 3  # future bins, after the scored bin
    np.testing.assert_array_equal(reference, extract(model, outside, indices, 1))
    for row in (5, 14):  # every bin inside the window can matter, including the last
        inside = counts.clone()
        inside[row] += 3
        assert not np.allclose(reference, extract(model, inside, indices, 1))


def test_masking_labels_and_views():
    torch.manual_seed(0)
    config = make_config(Hyperparameters())
    masker = Masker(config.TRAIN, torch.device("cpu"))
    spikes = torch.randint(1, 5, (64, BINS, UNITS))
    inputs, labels = masked(masker, spikes, False, 9)
    chosen = labels != UNMASKED_LABEL
    assert inputs.shape == labels.shape == spikes.shape
    assert .15 < chosen.float().mean() < .35  # upstream mask ratio 0.25
    assert (labels[chosen] == spikes[chosen]).all()
    assert (inputs[chosen] == 0).all() and (inputs[~chosen] == spikes[~chosen]).all()
    # Upstream caches mask probabilities per Masker, hence one Masker per purpose.
    view = masked(Masker(config.TRAIN, torch.device("cpu")), spikes, True, 9)[0]
    assert 0 < (view != spikes).float().mean() < .1  # contrast mask ratio 0.05


def test_schedule():
    assert warmup_cosine(0, 10, 100) == 0 and warmup_cosine(10, 10, 100) == 1
    assert warmup_cosine(5, 10, 100) == .5 and warmup_cosine(100, 10, 100) == 0
    assert abs(warmup_cosine(55, 10, 100) - .5) < 1e-12


def test_train_export_and_reload(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.5, (120, UNITS)).astype(np.int64)
    indices = np.arange(9, 120)[:, None] + np.arange(-9, 1)
    train = np.arange(len(indices)) < 70
    held_out = np.arange(len(indices)) >= 80
    hp = Hyperparameters(layers=2, hidden=16, epochs=1, batch_size=32)
    summary = train_and_export(counts, indices, tmp_path / "run", hp, train, held_out,
                               device="cpu")
    activation = np.load(tmp_path / "run/activations.npy")
    assert activation.shape == (len(indices), UNITS) and summary["complete"]
    assert summary["dimensions"] == UNITS and summary["reload_max_abs"] < 1e-5
    assert {"masked_nll", "mean_rate_nll", "bits_per_spike"} <= set(summary["validation"])
    with np.load(tmp_path / "run/index.npz") as index:
        np.testing.assert_array_equal(index["source_indices"], indices)
        np.testing.assert_array_equal(index["anchor_index"], indices[:, -1])
        np.testing.assert_array_equal(index["train_mask"], train)
    config = json.loads((tmp_path / "run/config.json").read_text())
    assert config["train_windows"] == 70 and config["validation_windows"] == 31
    restored = make_model(UNITS, BINS, hp, "cpu", config["max_spikes"])
    restored.load_state_dict(torch.load(tmp_path / "run/model_final.pt",
                                        weights_only=False)["state_dict"])
    values = torch.as_tensor(counts).clamp(max=config["max_spikes"])
    np.testing.assert_allclose(extract(restored, values, indices, 32), activation,
                               atol=1e-5)
    with pytest.raises(FileExistsError):  # frozen outputs are never overwritten
        train_and_export(counts, indices, tmp_path / "run", hp, device="cpu")
