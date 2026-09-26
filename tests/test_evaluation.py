"""Synthetic diagnostics: analytic contributions, nonlinear decoders, and known spectra."""

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nldisco.config import DecoderConfig, EncoderConfig, LossConfig, SedConfig
from nldisco.data.window import SpikeWindowDataset
from nldisco.evaluation import (
    AblationConfig,
    EvaluationConfig,
    SpectralConfig,
    ablation_squared_errors,
    evaluate_diagnostics,
    latent_ablation,
    save_diagnostics,
    spectral_comparison,
    spectral_from_model,
    validate_diagnostics,
)
from nldisco.model import build_sed
from nldisco.sweep.config import RunConfig
from nldisco.sweep.search import validate_config
from nldisco.sweep.training import run_trial, spectral_validation_loader
from nldisco.train import evaluate_model


def identity_model(seq_len=1):
    model = build_sed(
        SedConfig(
            n_neurons=2,
            seq_len=seq_len,
            dsed_topk_map={2: 2},
            decoder=DecoderConfig(output_activation="none"),
            inference_sparsity="sample_topk",
        )
    )
    with torch.no_grad():
        # FlatWindowEncoder exposes a single linear projection.
        for parameter in model.encoder.parameters():
            parameter.zero_()
        model.encoder.projection.weight.copy_(torch.eye(2).repeat(1, seq_len) / seq_len)
        model.decoder.weight.copy_(torch.eye(2).repeat(seq_len, 1, 1))
        model.decoder.bias.zero_()
    return model


def test_known_ablation_delta_r2_and_batch_invariance():
    model = identity_model()
    counts = torch.tensor([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0]])
    dataset = SpikeWindowDataset(counts, 1)
    first = latent_ablation(model, DataLoader(dataset, batch_size=1), AblationConfig(), [1.0])
    second = latent_ablation(model, DataLoader(dataset, batch_size=3), AblationConfig(), [1.0])
    pd.testing.assert_frame_equal(first, second)
    window = first[first.scope == "window"]
    # Each removal incurs SSE=8; total centered SST=8. Scores deliberately sum to 2.
    np.testing.assert_allclose(window.delta_r2, [1.0, 1.0])
    np.testing.assert_allclose(window.full_mse, 0)
    np.testing.assert_allclose(window.delta_mse, 1)
    assert model.training  # Standalone diagnostics restore mode.


@pytest.mark.parametrize("activation", ["none", "relu", "softplus"])
@pytest.mark.parametrize("temporal", [False, True])
def test_ablations_match_manual_decoder_with_bias_and_nonlinearity(activation, temporal):
    torch.manual_seed(8)
    model = build_sed(
        SedConfig(
            n_neurons=2,
            seq_len=4,
            dsed_topk_map={3: 2},
            encoder=EncoderConfig(
                type="TransformerWindow" if temporal else "FlatWindow",
                shift_equivariant=temporal,
                d_model=8,
                n_heads=2,
                n_layers=1,
                d_feedforward=16,
                attention_radius=1,
            ),
            decoder=DecoderConfig(output_activation=activation, temporal_kernel_len=2),
            inference_sparsity="sample_topk",
        )
    )
    with torch.no_grad():
        model.decoder.bias.fill_(-0.7)
    codes = torch.rand(5, 4, 3) if temporal else torch.rand(5, 3)
    targets = torch.randn(5, 4, 2)
    expected = []
    for j in [0, 2]:
        without = codes.clone()
        without[..., j] = 0
        prediction = model.decode(without)
        expected.append((prediction.double() - targets.double()).square().sum((0, 2)))
    actual = ablation_squared_errors(model, codes, targets, [0, 2], latent_batch_size=2)
    torch.testing.assert_close(actual, torch.stack(expected))


def test_redundant_harmful_and_inactive_latents():
    model = identity_model()
    with torch.no_grad():
        # Two identical codes both reconstruct the same unit, doubling its target.
        model.encoder.projection.weight.copy_(torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
        model.decoder.weight.copy_(torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]))
    loader = DataLoader(SpikeWindowDataset(torch.tensor([[1.0, 0.0], [2.0, 0.0]]), 1))
    table = latent_ablation(model, loader, AblationConfig(), [1.0])
    assert (table.delta_r2 < 0).all()
    with torch.no_grad():
        model.encoder.projection.weight[1].zero_()
    table = latent_ablation(model, loader, AblationConfig(latents=[1]), [1.0])
    assert (table.delta_mse == 0).all()


def test_weighted_sst_by_lag_and_constant_targets():
    model = identity_model(seq_len=2)
    counts = torch.tensor([[1.0, 0.0], [2.0, 1.0], [2.0, 0.0], [4.0, 3.0], [3.0, 0.0], [6.0, 5.0]])
    dataset = SpikeWindowDataset(counts, 2, stride=2)
    table = latent_ablation(
        model, DataLoader(dataset, batch_size=2), AblationConfig(latents=[0]), [1.0, 3.0]
    )
    lag = table[table.scope == "lag"]
    window = table[table.scope == "window"].iloc[0]
    assert window.delta_mse == pytest.approx(lag.delta_mse @ np.array([0.25, 0.75]))
    assert window.target_sst == pytest.approx(lag.target_sst @ np.array([0.25, 0.75]))
    assert window.delta_r2 == pytest.approx(window.delta_mse * 6 / window.target_sst)
    constant = DataLoader(SpikeWindowDataset(torch.ones(6, 2), 2))
    table = latent_ablation(model, constant, AblationConfig(), [1.0, 1.0])
    assert table.delta_r2.isna().all()
    assert np.isfinite(table.delta_mse).all()


def test_diagnostics_integration_level_selection_and_disabled_defaults():
    model = identity_model()
    loader = DataLoader(SpikeWindowDataset(torch.rand(8, 2), 1), batch_size=3)
    loss = LossConfig(type="mse", timebin_weights=[1.0])
    assert evaluate_model(model, loader, loss).diagnostics is None
    cfg = EvaluationConfig(ablation=AblationConfig(enabled=True, level=2, latents=[1]))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = evaluate_model(model, loader, loss, diagnostics=cfg)
    assert set(result.diagnostics.ablation.latent_idx) == {1}
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, equal_nan=True)
    cfg.ablation.level = 17
    with pytest.raises(ValueError, match="level"):
        evaluate_diagnostics(model, loader, cfg, [1.0])


def waves():
    t = np.arange(800) / 100
    low = np.sin(2 * np.pi * 5 * t)
    high = 0.5 * np.sin(2 * np.pi * 20 * t)
    return np.column_stack([low + high, low]), np.column_stack([low, low])


def test_known_frequency_loss_band_power_and_identity():
    target, reconstruction = waves()
    config = SpectralConfig(bin_size=0.01, segment_duration=1.0, bands={"high": [18.0, 22.0]})
    result = spectral_comparison(target, reconstruction, np.arange(len(target)), config)
    assert result.frequencies[np.argmax(result.target_psd[:, 0])] == 5
    assert result.metrics.relative_psd_error[0] == pytest.approx(0.2, abs=1e-10)
    assert result.metrics.relative_psd_error[1] == pytest.approx(0.0, abs=1e-10)
    high = result.band_power[result.band_power.unit_idx == 0].iloc[0]
    assert high.target_power == pytest.approx(0.125, abs=1e-10)
    assert high.reconstruction_power == pytest.approx(0.0, abs=1e-10)
    same = spectral_comparison(target, target.copy(), np.arange(len(target)), config)
    np.testing.assert_allclose(same.metrics.relative_psd_error, 0)


def test_psd_does_not_claim_phase_preservation():
    t = np.arange(800) / 100
    a, b = np.sin(2 * np.pi * 5 * t)[:, None], np.cos(2 * np.pi * 5 * t)[:, None]
    result = spectral_comparison(
        a, b, np.arange(len(a)), SpectralConfig(bin_size=0.01, segment_duration=1.0)
    )
    assert result.metrics.relative_psd_error[0] < 1e-10
    assert np.mean((a - b) ** 2) > 0.9


@pytest.mark.parametrize("boundary", ["source", "trial", "session", "timestamp"])
def test_spectra_do_not_cross_boundaries(boundary):
    # Both runs shorter than a segment: joining them would manufacture a spectrum.
    target = np.ones((60, 1))
    indices = np.arange(60)
    kwargs = {}
    if boundary == "source":
        indices[30:] += 5
    elif boundary == "timestamp":
        timestamps = np.arange(60) * 0.01
        timestamps[30:] += 1
        kwargs["timestamps"] = timestamps
    else:
        kwargs[f"{boundary}_codes"] = np.repeat([0, 1], 30)
    with pytest.raises(ValueError, match="No complete"):
        spectral_comparison(
            target, target, indices, SpectralConfig(bin_size=0.01, segment_duration=0.5), **kwargs
        )


def test_segment_weighting_order_duplicates_and_zero_power():
    target, _ = waves()
    trials = np.r_[np.zeros(200, dtype=int), np.ones(600, dtype=int)]
    config = SpectralConfig(bin_size=0.01, segment_duration=1.0)
    result = spectral_comparison(target, target, np.arange(800), config, trial_codes=trials)
    assert result.n_segments == 3 + 11
    order = np.random.default_rng(4).permutation(800)
    shuffled = spectral_comparison(
        target[order], target[order], np.arange(800)[order], config, trial_codes=trials[order]
    )
    np.testing.assert_allclose(result.target_psd, shuffled.target_psd)
    with pytest.raises(ValueError, match="once"):
        spectral_comparison(target, target, np.zeros(800, dtype=int), config)
    zeros = np.zeros_like(target)
    empty = spectral_comparison(zeros, zeros, np.arange(800), config)
    assert empty.metrics.relative_psd_error.isna().all()


def test_dense_spectral_loader_preserves_split_and_timestamp_gaps():
    counts = torch.ones(60, 2)
    times = np.arange(60) * 0.01
    times[30:] += 1.0
    dataset = SpikeWindowDataset(counts, 1, stride=5, timestamps=times, expected_bin_size=0.01)
    dense = spectral_validation_loader(DataLoader(dataset, batch_size=8))
    assert len(dense.dataset) == 60
    with pytest.raises(ValueError, match="No complete"):
        spectral_from_model(
            identity_model(), dense, SpectralConfig(bin_size=0.01, segment_duration=0.5)
        )


def test_hydra_trial_saves_diagnostics_and_preserves_training_stride(tmp_path):
    path = tmp_path / "activity.npy"
    np.save(path, np.random.default_rng(2).uniform(0, 2, (160, 2)).astype(np.float32))
    config = OmegaConf.structured(RunConfig)
    config.data.path = str(path)
    config.data.expected_bin_size = 0.01
    config.data.train_fraction = 0.5
    config.data.stride = 3
    config.model.seq_len = 3
    config.model.dsed_topk_map = {4: 2}
    config.loss.type = "mse"
    config.training.epochs = 1
    config.training.batch_size = 8
    config.evaluation = EvaluationConfig(
        ablation=AblationConfig(enabled=True, latents=[0, 1]),
        spectral=SpectralConfig(enabled=True, segment_duration=0.2, bands={"low": [0.0, 10.0]}),
    )
    validate_config(config)
    result = run_trial(OmegaConf.to_container(config), str(tmp_path / "trial"), "cpu")
    assert "validation/spectral/mean_relative_psd_error" in result
    outputs = tmp_path / "trial/diagnostics"
    assert {
        "latent_ablation.csv",
        "latent_ablation.pdf",
        "spectra.npz",
        "band_power.csv",
        "spectral_comparison.pdf",
        "spectral_metadata.json",
        "summary.json",
    } <= {p.name for p in outputs.iterdir()}
    saved = np.load(outputs / "spectra.npz")
    assert saved["target_psd"].shape == (11, 2)
    assert OmegaConf.load(tmp_path / "trial/config.yaml").data.stride == 3


def test_zero_power_outputs_save_without_nonfinite_json(tmp_path):
    zero = np.zeros((100, 2))
    config = SpectralConfig(bin_size=0.01, segment_duration=0.5)
    from nldisco.evaluation import DiagnosticResult

    result = DiagnosticResult(spectral=spectral_comparison(zero, zero, np.arange(100), config))
    summary = save_diagnostics(result, tmp_path / "diagnostics")
    assert "spectral/mean_relative_psd_error" not in summary


@pytest.mark.parametrize(
    "field,value",
    [
        ("bin_size", None),
        ("bin_size", float("nan")),
        ("bin_size", 0.0),
        ("segment_duration", 0.015),
        ("segment_duration", 0.0),
        ("lag", 1),
        ("bands", {"invalid": [0.0, 100.0]}),
    ],
)
def test_invalid_spectral_settings_fail_before_inference(field, value):
    options = EvaluationConfig(spectral=SpectralConfig(enabled=True, bin_size=0.01))
    setattr(options.spectral, field, value)
    with pytest.raises(ValueError):
        validate_diagnostics(options, identity_model().cfg)


def test_nested_level_and_integer_time_weights():
    model = build_sed(
        SedConfig(
            n_neurons=2,
            seq_len=1,
            dsed_topk_map={1: 1, 3: 2},
            inference_sparsity="sample_topk",
        )
    )
    loader = DataLoader(SpikeWindowDataset(torch.rand(8, 2), 1), batch_size=3)
    result = evaluate_diagnostics(
        model,
        loader,
        EvaluationConfig(
            ablation=AblationConfig(enabled=True, level=1),
        ),
        [1],
    )
    assert set(result.ablation.level) == {1}
    assert set(result.ablation.latent_idx) == {0}
    with pytest.raises(ValueError, match="indices"):
        latent_ablation(model, loader, AblationConfig(level=1, latents=[1]), [1])


@pytest.mark.parametrize("temporal", [False, True])
def test_full_diagnostics_match_evaluation_baseline_for_each_layout(temporal):
    model = build_sed(
        SedConfig(
            n_neurons=2,
            seq_len=4,
            dsed_topk_map={4: 2},
            encoder=EncoderConfig(
                type="TransformerWindow" if temporal else "FlatWindow",
                shift_equivariant=temporal,
                d_model=8,
                n_heads=2,
                n_layers=1,
                d_feedforward=16,
                attention_radius=1,
            ),
            decoder=DecoderConfig(output_activation="softplus", temporal_kernel_len=2),
            inference_sparsity="sample_topk",
        )
    )
    counts = torch.rand(80, 2)
    dataset = SpikeWindowDataset(
        counts, 4, stride=3, occurrence_support=(1, 0) if temporal else None
    )
    loader = DataLoader(dataset, batch_size=8)
    options = EvaluationConfig(
        ablation=AblationConfig(enabled=True, latents=[0]),
        spectral=SpectralConfig(enabled=True, bin_size=0.01, segment_duration=0.2, lag=-1),
    )
    result = evaluate_model(
        model,
        loader,
        LossConfig(type="mse", timebin_weights=[1] * 4),
        diagnostics=options,
        spectral_loader=spectral_validation_loader(loader),
    )
    assert result.diagnostics.ablation.query("scope == 'window'").full_mse.iloc[
        0
    ] == pytest.approx(result.weighted_reconstruction, rel=1e-5)
    assert result.diagnostics.spectral.lag == -1


def test_wandb_diagnostic_tables_are_logged_only_when_enabled(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    path = tmp_path / "activity.npy"
    np.save(path, np.random.default_rng(1).uniform(size=(80, 2)).astype(np.float32))
    config = OmegaConf.structured(RunConfig)
    config.data.path = str(path)
    config.model.seq_len = 1
    config.model.dsed_topk_map = {4: 2}
    config.loss.type = "mse"
    config.training.epochs = 1
    config.wandb.enabled = True
    config.evaluation.ablation.enabled = True
    config.evaluation.spectral.enabled = True
    config.evaluation.spectral.bin_size = 0.01
    config.evaluation.spectral.segment_duration = 0.08
    logger, table = MagicMock(), MagicMock()
    monkeypatch.setattr("wandb.log", logger)
    monkeypatch.setattr("wandb.Table", table)
    run_trial(OmegaConf.to_container(config), str(tmp_path / "trial"), "cpu")
    keys = {key for call in logger.call_args_list for key in call.args[0]}
    assert {"evaluation/latent_ablation", "evaluation/spectral_metrics"} <= keys
    assert table.call_count == 2
