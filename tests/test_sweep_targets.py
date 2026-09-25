"""Paired-population preprocessing, configuration, and portable checkpoints."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from nldisco.model import build_sed
from nldisco.sweep.config import DATA_PATH_FIELDS, RunConfig, register_configs
from nldisco.sweep.main import launch
from nldisco.sweep.search import validate_config
from nldisco.sweep.training import (
    load_checkpoint,
    model_and_loss,
    prepare_data,
    run_trial,
    spectral_validation_loader,
)


@pytest.fixture
def paired_config(tmp_path):
    rng = np.random.default_rng(82)
    inputs = rng.normal(size=(96, 3)).astype(np.float32)
    targets = rng.poisson(2, size=(96, 5)).astype(np.float32)
    np.save(tmp_path / "inputs.npy", inputs)
    np.save(tmp_path / "targets.npy", targets)
    cfg = OmegaConf.structured(RunConfig)
    cfg.data.path = str(tmp_path / "inputs.npy")
    cfg.data.target_path = str(tmp_path / "targets.npy")
    cfg.data.input_population = "input-region"
    cfg.data.target_population = "output-region"
    cfg.output_dir = str(tmp_path / "outputs")
    cfg.env_file = None
    cfg.model.seq_len = 4
    cfg.model.dsed_topk_map = {4: 1, 8: 2}
    cfg.model.encoder.d_model = 8
    cfg.model.encoder.n_heads = 2
    cfg.model.encoder.n_layers = 1
    cfg.model.encoder.d_feedforward = 16
    cfg.model.encoder.attention_radius = 1
    cfg.model.decoder.temporal_kernel_len = 2
    cfg.model.decoder.output_activation = "none"
    cfg.training.epochs = 1
    cfg.training.batch_size = 8
    cfg.loss.type = "mse"
    return cfg


def _plain(cfg):
    return OmegaConf.to_container(cfg, resolve=True)


def _save_metadata(cfg, name, values, tmp_path):
    path = tmp_path / f"{name}.npy"
    np.save(path, values)
    cfg.data[name] = str(path)


def test_target_hydra_fields():
    register_configs()
    with initialize_config_module(version_base="1.3", config_module="nldisco.sweep.conf"):
        cfg = compose(config_name="train", overrides=[
            "data.target_path=targets.npy", "data.target_normalization=zscore",
            "data.target_valid_rows=valid.npy", "data.target_population=region-b",
        ])
    assert cfg.data.target_path == "targets.npy"
    assert cfg.data.target_normalization == "zscore"
    assert cfg.data.target_valid_rows == "valid.npy"
    assert cfg.data.target_population == "region-b"


def test_independent_train_only_statistics_and_joint_validity(paired_config, tmp_path):
    cfg = paired_config
    inputs = np.arange(288, dtype=np.float32).reshape(96, 3)
    targets = 100 + np.arange(480, dtype=np.float32).reshape(96, 5) * 3
    inputs[20] = np.nan
    targets[30] = np.nan
    np.save(cfg.data.path, inputs)
    np.save(cfg.data.target_path, targets)
    for name, invalid in (("valid_rows", 20), ("target_valid_rows", 30)):
        valid = np.ones(96, dtype=bool)
        valid[invalid] = False
        _save_metadata(cfg, name, valid, tmp_path)
    cfg.data.normalization = "zscore"
    cfg.data.target_normalization = "minmax"
    model, _ = model_and_loss(cfg, 3, 5)
    train, val, input_stats, target_stats = prepare_data(cfg, model, return_target_stats=True)
    fitting = np.arange(96) < int(96 * cfg.data.train_fraction)
    fitting[[20, 30]] = False
    np.testing.assert_allclose(input_stats.offset, inputs[fitting].mean(0))
    np.testing.assert_allclose(target_stats.offset, targets[fitting].min(0))
    np.testing.assert_allclose(target_stats.scale, np.ptp(targets[fitting], axis=0))
    training_rows = set(torch.cat([sample.source_indices for sample in train.dataset]).tolist())
    validation_rows = set(torch.cat([sample.source_indices for sample in val.dataset]).tolist())
    assert not training_rows.intersection(validation_rows)
    assert not {20, 30}.intersection(training_rows | validation_rows)
    sample = val.dataset[0]
    np.testing.assert_allclose(sample.values, input_stats.transform(inputs[sample.source_indices]), rtol=1e-6)
    np.testing.assert_allclose(sample.target, target_stats.transform(targets[sample.source_indices]), rtol=1e-6)
    assert sample.target.min() > 1  # Held-out values are transformed, never clipped/refitted.
    batch = next(iter(train))
    assert batch.values.shape[-1] == 3 and batch.target.shape[-1] == 5
    dense = spectral_validation_loader(val)
    assert dense.dataset.targets is val.dataset.targets
    assert dense.dataset.spike_counts is val.dataset.spike_counts
    assert all(set(s.source_indices.tolist()) <= validation_rows for s in dense.dataset)


@pytest.mark.parametrize("split", ["trial", "session"])
def test_target_metadata_controls_shared_boundaries(paired_config, tmp_path, split):
    cfg = paired_config
    groups = np.repeat(np.arange(4), 24)
    _save_metadata(cfg, "target_trial_ids", groups, tmp_path)
    _save_metadata(cfg, "target_session_ids", groups, tmp_path)
    _save_metadata(cfg, "target_timestamps", np.arange(96, dtype=float), tmp_path)
    cfg.data.expected_bin_size = 1.0
    cfg.data.split = split
    cfg.data.train_sessions = ["0", "1"]
    model, _ = model_and_loss(cfg, 3, 5)
    train, val, _ = prepare_data(cfg, model)
    train_groups = {groups[s.source_indices[0]] for s in train.dataset}
    val_groups = {groups[s.source_indices[0]] for s in val.dataset}
    assert train_groups and val_groups and not train_groups.intersection(val_groups)
    targets = np.load(cfg.data.target_path)
    for sample in list(train.dataset) + list(val.dataset):
        assert len(set(groups[sample.source_indices])) == 1
        np.testing.assert_array_equal(sample.target, targets[sample.source_indices])


@pytest.mark.parametrize("name", ["trial_ids", "session_ids", "timestamps"])
def test_misaligned_target_metadata_rejected(paired_config, tmp_path, name):
    values = np.arange(96)
    _save_metadata(paired_config, name, values, tmp_path)
    _save_metadata(paired_config, f"target_{name}", values + 1, tmp_path)
    paired_config.data.expected_bin_size = 1.0
    model, _ = model_and_loss(paired_config, 3, 5)
    with pytest.raises(ValueError, match="align"):
        prepare_data(paired_config, model)


@pytest.mark.parametrize("loss", ["mse", "msle"])
def test_zscored_inputs_can_predict_nonnegative_targets(paired_config, loss):
    cfg = paired_config
    cfg.data.normalization = "zscore"
    cfg.loss.type = loss
    cfg.model.decoder.output_activation = "relu"
    validate_config(cfg)
    model, _ = model_and_loss(cfg, 3, 5)
    train, _, input_stats, target_stats = prepare_data(cfg, model, return_target_stats=True)
    assert input_stats is not None and target_stats is None
    assert train.dataset.spike_counts.min() < 0
    assert train.dataset.targets.min() >= 0


@pytest.mark.parametrize("loss,activation", [("msle", "none"), ("mse", "relu")])
def test_zscored_target_domain_is_validated(paired_config, loss, activation):
    cfg = paired_config
    cfg.data.target_normalization = "zscore"
    cfg.loss.type = loss
    cfg.model.decoder.output_activation = activation
    with pytest.raises(ValueError, match="Target zscore requires"):
        validate_config(cfg)


@pytest.mark.parametrize("loss,activation", [("msle", "relu"), ("mse", "relu")])
def test_negative_heldout_targets_rejected(paired_config, loss, activation):
    cfg = paired_config
    targets = np.ones((96, 5), dtype=np.float32)
    targets[80:] = -1
    np.save(cfg.data.target_path, targets)
    cfg.loss.type = loss
    cfg.model.decoder.output_activation = activation
    model, _ = model_and_loss(cfg, 3, 5)
    with pytest.raises(ValueError, match="targets"):
        prepare_data(cfg, model)


@pytest.mark.parametrize("which,shape", [("input", (96, 4)), ("target", (96, 4)), ("target", (95, 5))])
def test_incompatible_shapes_rejected_before_training(paired_config, which, shape):
    cfg = paired_config
    np.save(cfg.data.path if which == "input" else cfg.data.target_path, np.ones(shape))
    model, _ = model_and_loss(cfg, 3, 5)
    with pytest.raises(ValueError, match="width|aligned"):
        prepare_data(cfg, model)


@pytest.mark.parametrize("method", ["none", "zscore", "minmax"])
def test_explicit_equal_targets_preserve_autoencoder(paired_config, method):
    cfg = paired_config
    cfg.data.target_path = cfg.data.path
    cfg.data.normalization = cfg.data.target_normalization = method
    cfg.model.decoder.output_activation = "none"
    model, _ = model_and_loss(cfg, 3, 3)
    paired_train, paired_val, stats, target_stats = prepare_data(cfg, model, return_target_stats=True)
    cfg.data.target_path = None
    auto_train, auto_val, auto_stats, auto_target_stats = prepare_data(cfg, model, return_target_stats=True)
    assert auto_stats is auto_target_stats
    for paired, auto in ((paired_train, auto_train), (paired_val, auto_val)):
        assert paired.dataset.valid_starts == auto.dataset.valid_starts
        for paired_sample, auto_sample in zip(paired.dataset, auto.dataset):
            torch.testing.assert_close(paired_sample.values, auto_sample.values)
            torch.testing.assert_close(paired_sample.target, auto_sample.target)
    if stats is not None:
        np.testing.assert_array_equal(stats.offset, target_stats.offset)
        np.testing.assert_array_equal(stats.scale, auto_stats.scale)


@pytest.mark.parametrize("architecture", ["single", "flat", "transformer", "temporal"])
@pytest.mark.parametrize("n_output", [3, 5])
def test_paired_trial_and_portable_input_only_reload(paired_config, tmp_path, monkeypatch, architecture, n_output):
    cfg = paired_config
    targets = np.load(cfg.data.target_path)[:, :n_output]
    np.save(cfg.data.target_path, targets)
    cfg.data.normalization = "zscore"
    cfg.data.target_normalization = "zscore"
    if architecture == "single":
        cfg.model.seq_len = 1
    elif architecture in {"transformer", "temporal"}:
        cfg.model.encoder.type = "TransformerWindow"
        cfg.model.encoder.shift_equivariant = architecture == "temporal"
    output = tmp_path / "trial"
    result = run_trial(_plain(cfg), str(output), "cpu")
    assert result["status"] == "completed"
    assert np.isfinite(result["validation/loss"])
    checkpoint = torch.load(output / "model.pt", weights_only=True)
    assert checkpoint["n_units"] == checkpoint["n_input_units"] == 3
    assert checkpoint["n_output_units"] == n_output
    assert checkpoint["config"]["data"]["target_population"] == "output-region"
    assert (output / "normalization.npz").is_file()
    assert (output / "target_normalization.npz").is_file()
    portable = tmp_path / "portable.pt"
    torch.save(checkpoint, portable)
    # A self-contained checkpoint must not open data files or normalization sidecars.
    monkeypatch.setattr(np, "load", MagicMock(side_effect=AssertionError("unexpected data access")))
    restored = load_checkpoint(portable)
    assert restored.model.cfg.n_neurons == 3
    assert restored.model.cfg.n_output_neurons == n_output
    assert not restored.model.training
    assert restored.input_normalization.offset.shape == (3,)
    assert restored.target_normalization.offset.shape == (n_output,)
    model_cfg, _ = model_and_loss(cfg, 3, n_output)
    original = build_sed(model_cfg).eval()
    original.load_state_dict(checkpoint["state_dict"])
    x = torch.randn(2, cfg.model.seq_len, 3)
    with torch.no_grad():
        expected = original(x).reconstructions
        actual = restored.model(x).reconstructions
    for level in expected:
        assert actual[level].shape == (2, cfg.model.seq_len, n_output)
        torch.testing.assert_close(actual[level], expected[level])


@pytest.mark.parametrize("normalization", ["none", "zscore"])
def test_legacy_autoencoder_checkpoint(paired_config, tmp_path, normalization):
    cfg = paired_config
    cfg.data.target_path = None
    cfg.data.normalization = normalization
    config = _plain(cfg)
    config["data"] = {key: value for key, value in config["data"].items() if not key.startswith("target_")}
    model_cfg, _ = model_and_loss(OmegaConf.create(config), 3)
    model = build_sed(model_cfg)
    torch.save({"config": config, "n_units": 3, "state_dict": model.state_dict()}, tmp_path / "model.pt")
    if normalization == "zscore":
        np.savez(tmp_path / "normalization.npz", method="zscore", offset=np.arange(3), scale=np.ones(3))
    saved = load_checkpoint(tmp_path / "model.pt")
    assert saved.model.cfg.n_output_neurons == 3
    if normalization == "none":
        assert saved.input_normalization is saved.target_normalization is None
    else:
        np.testing.assert_array_equal(saved.input_normalization.offset, np.arange(3))
        np.testing.assert_array_equal(saved.target_normalization.offset, np.arange(3))


def test_checkpoint_rejects_wrong_normalizer_width(paired_config, tmp_path):
    model_cfg, _ = model_and_loss(paired_config, 3, 5)
    paired_config.data.target_normalization = "zscore"
    torch.save({
        "config": _plain(paired_config), "n_input_units": 3, "n_output_units": 5,
        "state_dict": build_sed(model_cfg).state_dict(),
        "normalization": {"input": None, "target": {
            "method": "zscore", "offset": torch.zeros(3), "scale": torch.ones(3),
        }},
    }, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="normalization statistics"):
        load_checkpoint(tmp_path / "bad.pt")


def test_target_paths_resolved_before_dispatch(paired_config, tmp_path, monkeypatch):
    cfg = paired_config
    _save_metadata(cfg, "target_valid_rows", np.ones(96, dtype=bool), tmp_path)
    _save_metadata(cfg, "target_timestamps", np.arange(96, dtype=float), tmp_path)
    cfg.data.expected_bin_size = 1.0
    monkeypatch.chdir(tmp_path)
    for key in DATA_PATH_FIELDS:
        if cfg.data[key] is not None:
            cfg.data[key] = Path(cfg.data[key]).name
    dispatch = MagicMock(return_value={"status": "submitted"})
    monkeypatch.setattr("nldisco.sweep.main.execute", dispatch)
    launch(cfg)
    resolved = dispatch.call_args.args[1]["data"]
    assert resolved["target_path"] == str(tmp_path / "targets.npy")
    assert resolved["target_valid_rows"] == str(tmp_path / "target_valid_rows.npy")
    assert resolved["target_timestamps"] == str(tmp_path / "target_timestamps.npy")


def test_missing_target_file_checked_before_dispatch(paired_config, monkeypatch):
    paired_config.data.target_path += ".missing"
    dispatch = MagicMock()
    monkeypatch.setattr("nldisco.sweep.main.execute", dispatch)
    with pytest.raises(FileNotFoundError):
        launch(paired_config)
    dispatch.assert_not_called()


def test_target_config_validation(paired_config):
    cfg = paired_config
    cfg.data.target_normalization = "invalid"
    with pytest.raises(ValueError, match="target_normalization"):
        validate_config(cfg)
    cfg.data.target_normalization = "zscore"
    cfg.data.target_path = None
    with pytest.raises(ValueError, match="target normalization requires"):
        validate_config(cfg)
    cfg.data.target_normalization = None
    cfg.data.target_valid_rows = "valid.npy"
    with pytest.raises(ValueError, match="Target row metadata requires"):
        validate_config(cfg)
