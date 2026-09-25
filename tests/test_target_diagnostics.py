"""Paired-population diagnostics always encode inputs and score output units."""

import json

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from nldisco.config import DecoderConfig, EncoderConfig, LossConfig, SedConfig
from nldisco.data.window import SpikeWindowDataset
from nldisco.evaluation import (
    AblationConfig,
    EvaluationConfig,
    SpectralConfig,
    latent_ablation,
    save_diagnostics,
    spectral_comparison,
    spectral_from_model,
)
from nldisco.model import build_sed
from nldisco.plot import decoder_feature_matrix, plot_decoder_feature, plot_spectral_comparison
from nldisco.sweep.training import spectral_validation_loader
from nldisco.train import evaluate_model


def paired_model(layout, n_output):
    temporal = layout == "shift_equivariant"
    return build_sed(
        SedConfig(
            n_neurons=2,
            n_output_neurons=n_output,
            seq_len=1 if layout == "single_bin" else 4,
            dsed_topk_map={2: 1, 4: 2},
            encoder=EncoderConfig(
                type="TransformerWindow"
                if layout in {"transformer", "shift_equivariant"}
                else "FlatWindow",
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


@pytest.mark.parametrize("n_output", [2, 3])
@pytest.mark.parametrize("layout", ["single_bin", "flat", "transformer", "shift_equivariant"])
def test_paired_diagnostics_match_manual_output_population(layout, n_output):
    torch.manual_seed(45)
    model = paired_model(layout, n_output)
    seq_len = model.cfg.seq_len
    inputs = torch.rand(40, 2)
    targets = inputs @ torch.rand(2, n_output) + 2
    temporal = model.cfg.encoder.shift_equivariant
    dataset = SpikeWindowDataset(
        inputs,
        seq_len,
        stride=model.cfg.usable_occurrence_positions if temporal else seq_len,
        targets=targets,
        occurrence_support=model.cfg.temporal_occurrence_support() if temporal else None,
    )
    loader = DataLoader(dataset, batch_size=64)
    config = EvaluationConfig(
        ablation=AblationConfig(enabled=True, level=2, latents=[0]),
        spectral=SpectralConfig(enabled=True, bin_size=0.01, segment_duration=0.08),
    )
    dense_loader = spectral_validation_loader(loader)
    result = evaluate_model(
        model,
        loader,
        LossConfig(type="mse", timebin_weights=[1] * seq_len),
        diagnostics=config,
        spectral_loader=dense_loader,
    )
    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        output = model(batch.values)
        expected_loss = (output.reconstructions[2] - batch.targets).square().mean().item()
        largest_loss = (output.reconstructions[4] - batch.targets).square().mean().item()
    actual = result.diagnostics.ablation.query("scope == 'window'").full_mse.iloc[0]
    assert actual == pytest.approx(expected_loss, rel=1e-6)
    assert result.weighted_reconstruction == pytest.approx(largest_loss, rel=1e-6)

    assert torch.equal(dense_loader.dataset.targets, targets)
    selected = np.arange(seq_len - 1, len(targets))
    reference = spectral_comparison(
        targets[selected].numpy(), targets[selected].numpy(), selected, config.spectral
    )
    np.testing.assert_allclose(result.diagnostics.spectral.target_psd, reference.target_psd)
    assert len(result.diagnostics.spectral.metrics) == n_output
    assert decoder_feature_matrix(model.decoder, 0).shape[1] == n_output


def test_ablation_denominator_and_target_variance_are_in_output_space():
    model = build_sed(
        SedConfig(
            n_neurons=2,
            n_output_neurons=3,
            seq_len=1,
            dsed_topk_map={2: 2},
            decoder=DecoderConfig(output_activation="none"),
            inference_sparsity="sample_topk",
        )
    )
    transform = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    with torch.no_grad():
        model.encoder.projection.weight.copy_(torch.eye(2))
        model.encoder.projection.bias.zero_()
        model.decoder.weight.copy_(transform[None])
        model.decoder.bias.zero_()
    inputs = torch.tensor([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]])
    targets = inputs @ transform.T + 0.5
    loader = DataLoader(SpikeWindowDataset(inputs, 1, targets=targets), batch_size=3)
    result = latent_ablation(model, loader, AblationConfig(latents=[0]), [1])
    expected = inputs @ transform.T
    ablated = inputs[:, 1:] @ transform[:, 1:].T
    full_sse = (targets - expected).square().sum().item()
    ablated_sse = (targets - ablated).square().sum().item()
    sst = (targets - targets.mean(0)).square().sum().item()
    np.testing.assert_allclose(result.full_mse, full_sse / targets.numel())
    np.testing.assert_allclose(result.ablated_mse, ablated_sse / targets.numel())
    np.testing.assert_allclose(result.delta_r2, (ablated_sse - full_sse) / sst)
    np.testing.assert_allclose(result.target_sst, sst)


def test_paired_dense_loader_retains_split_boundaries_and_target_values():
    inputs = torch.rand(48, 2)
    targets = 4 * torch.rand(48, 3)
    allowed = np.zeros(48, dtype=bool)
    allowed[24:] = True
    target_valid = np.ones(48, dtype=bool)
    target_valid[36] = False
    dataset = SpikeWindowDataset(
        inputs,
        4,
        stride=4,
        targets=targets,
        allowed_rows=allowed,
        target_valid_rows=target_valid,
    )
    dense = spectral_validation_loader(DataLoader(dataset, batch_size=8))
    assert dense.dataset.valid_starts == [
        start for start in range(24, 45) if not start <= 36 < start + 4
    ]
    for batch in dense:
        assert (batch.source_indices >= 24).all()
        assert (batch.source_indices != 36).all()
        torch.testing.assert_close(batch.targets, targets[batch.source_indices])
        torch.testing.assert_close(batch.values, inputs[batch.source_indices])


def test_explicit_input_targets_preserve_autoencoder_diagnostics():
    model = paired_model("flat", 2)
    inputs = torch.rand(40, 2)
    implicit = DataLoader(SpikeWindowDataset(inputs, 4), batch_size=13)
    explicit = DataLoader(SpikeWindowDataset(inputs, 4, targets=inputs.clone()), batch_size=13)
    ablation = AblationConfig(latents=[0, 1])
    assert latent_ablation(model, implicit, ablation, [1] * 4).equals(
        latent_ablation(model, explicit, ablation, [1] * 4)
    )
    config = SpectralConfig(bin_size=0.01, segment_duration=0.08)
    first = spectral_from_model(model, implicit, config)
    second = spectral_from_model(model, explicit, config)
    np.testing.assert_array_equal(first.target_psd, second.target_psd)
    np.testing.assert_array_equal(first.reconstruction_psd, second.reconstruction_psd)


@pytest.mark.parametrize("diagnostic", ["ablation", "spectral"])
def test_diagnostics_reject_broadcastable_target_width_before_encoding(diagnostic):
    model = paired_model("single_bin", 3)
    loader = DataLoader(SpikeWindowDataset(torch.rand(16, 2), 1, targets=torch.rand(16, 1)))
    with pytest.raises(ValueError, match="targets must have shape"):
        if diagnostic == "ablation":
            latent_ablation(model, loader, AblationConfig(), [1])
        else:
            spectral_from_model(
                model, loader, SpectralConfig(bin_size=0.01, segment_duration=0.08)
            )
    assert model.training


def test_saved_spectra_and_decoder_plots_identify_output_population(tmp_path):
    from nldisco.evaluation import DiagnosticResult

    target = np.arange(32, dtype=float).reshape(16, 2)
    spectrum = spectral_comparison(
        target,
        target.copy(),
        np.arange(16),
        SpectralConfig(bin_size=0.01, segment_duration=0.08),
    )
    save_diagnostics(DiagnosticResult(spectral=spectrum), tmp_path / "diagnostics")
    metadata = json.loads((tmp_path / "diagnostics/spectral_metadata.json").read_text())
    assert metadata["population"] == "output"
    assert metadata["n_output_units"] == 2
    assert metadata["units"] == "model target units squared per Hz"
    decoder_ax = plot_decoder_feature(paired_model("single_bin", 3).decoder, 0)
    assert decoder_ax.get_ylabel() == "Output unit"
    spectrum_ax = plot_spectral_comparison(spectrum)
    assert "target" in spectrum_ax.get_ylabel()
    plt.close(decoder_ax.figure)
    plt.close(spectrum_ax.figure)
