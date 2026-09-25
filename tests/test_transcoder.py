"""End-to-end regressions for optional aligned transcoder supervision."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import r2_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, default_collate

from nldisco.config import DecoderConfig, EncoderConfig, LossConfig, SedConfig, TrainConfig
from nldisco.data.window import SpikeWindowDataset, WindowSample
from nldisco.model import build_sed
from nldisco.train import evaluate_model, train_model, weighted_reconstruction_loss


def _config(architecture, n_output=2, **kwargs):
    seq_len = 1 if architecture == "single" else 4
    encoder = EncoderConfig() if architecture in {"single", "flat"} else EncoderConfig(
        type="TransformerWindow", shift_equivariant=architecture == "shift",
        d_model=8, n_heads=2, n_layers=1, d_feedforward=12, attention_radius=1,
    )
    return SedConfig(
        n_neurons=3, n_output_neurons=n_output, seq_len=seq_len,
        dsed_topk_map={3: 1, 5: 2}, encoder=encoder,
        decoder=DecoderConfig(output_activation="none", temporal_kernel_len=2),
        inference_sparsity="sample_topk", **kwargs,
    )


def _loader(cfg, inputs, targets=None):
    support = cfg.temporal_occurrence_support() if cfg.encoder.shift_equivariant else None
    dataset = SpikeWindowDataset(
        inputs, cfg.seq_len, stride=cfg.seq_len if support is None else cfg.usable_occurrence_positions,
        targets=targets, occurrence_support=support,
    )
    return DataLoader(dataset, batch_size=4)


@pytest.mark.parametrize("architecture", ["single", "flat", "transformer", "shift"])
@pytest.mark.parametrize("n_output", [2, 3])
def test_transcoder_trains_all_levels_and_evaluates_targets(architecture, n_output):
    torch.manual_seed(9)
    cfg = _config(architecture, n_output)
    inputs = torch.randn(8, 3)
    targets = torch.arange(8 * n_output).reshape(8, n_output).float() / 8 + 3
    loader = _loader(cfg, inputs, targets)
    model = build_sed(cfg)
    before = copy.deepcopy(model.state_dict())
    loss = LossConfig(type="mse", timebin_weights=[1] * cfg.seq_len)
    train_model(model, loader, loss, TrainConfig(epochs=1, use_lr_schedule=False))
    assert not torch.equal(before["decoder.weight"], model.state_dict()["decoder.weight"])
    result = evaluate_model(model, loader, loss)
    expected_targets = torch.stack([sample.targets for sample in loader.dataset])
    torch.testing.assert_close(result.targets, expected_targets)
    expected_loss, per_lag = weighted_reconstruction_loss(
        expected_targets, result.reconstructions, loss
    )
    assert result.weighted_reconstruction == pytest.approx(expected_loss.item())
    assert result.metrics_by_lag.reconstruction_loss.tolist() == pytest.approx(per_lag.tolist())
    assert result.metrics_by_lag.cosine_similarity.iloc[-1] == pytest.approx(
        F.cosine_similarity(result.reconstructions[:, -1], expected_targets[:, -1]).mean().item()
    )
    assert result.metrics_by_lag.r2.iloc[-1] == pytest.approx(r2_score(
        expected_targets[:, -1], result.reconstructions[:, -1], multioutput="variance_weighted"
    ))
    # Prediction requires inputs alone, including with an independently sized decoder.
    output = model(inputs[:cfg.seq_len].unsqueeze(0))
    assert {k: v.shape for k, v in output.reconstructions.items()} == {
        3: (1, cfg.seq_len, n_output), 5: (1, cfg.seq_len, n_output)
    }


@pytest.mark.parametrize("architecture", ["single", "flat", "transformer", "shift"])
def test_explicit_identical_targets_preserve_autoencoder_training(architecture):
    torch.manual_seed(13)
    cfg = _config(architecture, 3)
    values = torch.rand(8, 3)
    implicit = build_sed(cfg)
    explicit = copy.deepcopy(implicit)
    loss = LossConfig(type="mse", timebin_weights=[1] * cfg.seq_len)
    training = TrainConfig(epochs=1, use_lr_schedule=False)
    first = train_model(implicit, _loader(cfg, values), loss, training)
    second = train_model(explicit, _loader(cfg, values, values.clone()), loss, training)
    assert first == second
    for key, value in implicit.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], equal_nan=True)


def test_sample_legacy_constructor_and_paired_collation():
    original = SpikeWindowDataset(torch.ones(3, 2), 2)[0]
    legacy = WindowSample(*original[:7])
    assert legacy.targets is legacy.values
    assert legacy.target is legacy.values
    collated = default_collate([legacy, legacy])
    assert isinstance(collated, WindowSample)
    torch.testing.assert_close(collated.values, collated.targets)
    paired = SpikeWindowDataset(torch.ones(3, 2), 2, targets=torch.full((3, 4), 9.0))
    batch = next(iter(DataLoader(paired, batch_size=2)))
    assert batch.values.shape == (2, 2, 2)
    assert batch.targets.shape == (2, 2, 4)


def test_paired_windows_share_boundaries_and_both_validity_masks():
    inputs = torch.arange(24).reshape(12, 2).float()
    targets = torch.arange(36).reshape(12, 3).float()
    inputs[1, 0] = float("nan")
    targets[9, 0] = float("inf")
    allowed = np.ones(12, dtype=bool)
    allowed[5] = False
    target_valid = np.ones(12, dtype=bool)
    target_valid[10] = False
    dataset = SpikeWindowDataset(
        inputs, 2, targets=targets, allowed_rows=allowed, target_valid_rows=target_valid,
        trial_ids=[0] * 3 + [1] * 4 + [2] * 5,
        session_ids=[0] * 4 + [1] * 8,
        timestamps=np.array([0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14.0]),
        expected_bin_size=1,
    )
    assert dataset.valid_starts == [7]
    assert not allowed[5]
    assert allowed[1]  # Caller-owned split mask was not changed by invalid rows.
    torch.testing.assert_close(dataset[0].values, inputs[7:9])
    torch.testing.assert_close(dataset[0].targets, targets[7:9])
    assert dataset[0].source_indices.tolist() == [7, 8]


@pytest.mark.parametrize("metadata", ["source_indices", "trial_ids", "session_ids", "timestamps"])
def test_target_metadata_must_be_aligned(metadata):
    kwargs = {metadata: [0, 1, 2], "target_" + metadata: [0, 2, 3]}
    if metadata == "timestamps":
        kwargs["expected_bin_size"] = 1
    with pytest.raises(ValueError, match="temporally aligned"):
        SpikeWindowDataset(torch.ones(3, 2), 1, targets=torch.ones(3, 4), **kwargs)


def test_single_bin_missing_timestamps_and_nullable_groups_are_invalid():
    dataset = SpikeWindowDataset(
        torch.ones(3, 2), 1, targets=torch.ones(3, 4), timestamps=[0, float("nan"), 2],
        expected_bin_size=1, trial_ids=[1, pd.NA, 1], target_trial_ids=[1, pd.NA, 1],
    )
    assert dataset.valid_starts == [0, 2]


@pytest.mark.parametrize("metadata", ["trial_ids", "session_ids", "timestamps"])
def test_missing_metadata_on_either_side_invalidates_shared_rows(metadata):
    shared = [1, 1, np.nan, 1]
    target = [1, np.nan, 1, 1]
    kwargs = {metadata: shared, "target_" + metadata: target}
    if metadata == "timestamps":
        kwargs["expected_bin_size"] = 1
    dataset = SpikeWindowDataset(torch.ones(4, 2), 1, targets=torch.ones(4, 3), **kwargs)
    assert dataset.valid_starts == [0, 3]
    assert dataset.allowed_rows.tolist() == [True, False, False, True]


@pytest.mark.parametrize("shape", [(3, 2, 1), (4, 2), (3, 0)])
def test_dataset_rejects_incompatible_target_time_axes(shape):
    with pytest.raises(ValueError, match="targets must have shape"):
        SpikeWindowDataset(torch.ones(3, 2), 1, targets=torch.ones(shape))


def test_target_shape_validation_precedes_training():
    cfg = _config("single", 4)
    model = build_sed(cfg)
    before = copy.deepcopy(model.state_dict())
    loader = _loader(cfg, torch.ones(4, 3))
    with pytest.raises(ValueError, match="targets must have shape"):
        train_model(model, loader, LossConfig(type="mse", timebin_weights=[1]), TrainConfig())
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, equal_nan=True)


def test_target_domain_checks_allow_signed_inputs_for_msle():
    cfg = _config("single", 2)
    cfg.decoder.output_activation = "relu"
    inputs = -torch.ones(4, 3)
    targets = torch.ones(4, 2)
    loss = LossConfig(type="msle", timebin_weights=[1])
    train_model(build_sed(cfg), _loader(cfg, inputs, targets), loss, TrainConfig(epochs=1))
    with pytest.raises(ValueError, match="nonnegative targets"):
        train_model(build_sed(cfg), _loader(cfg, inputs, -targets), loss, TrainConfig(epochs=1))
    with pytest.raises(ValueError, match="Signed targets"):
        train_model(
            build_sed(cfg), _loader(cfg, inputs, -targets),
            LossConfig(type="mse", timebin_weights=[1]), TrainConfig(epochs=1),
        )


def test_matryoshka_loss_compares_every_level_to_distinct_targets():
    cfg = _config("single", 3)
    model = build_sed(cfg)
    inputs = torch.ones(4, 3)
    targets = torch.full((4, 3), 7.0)
    loader = _loader(cfg, inputs, targets)
    loss = LossConfig(type="mse", timebin_weights=[1], dsed_loss_weight_map={3: 0.25, 5: 2})
    output = model(inputs.unsqueeze(1))
    expected = sum(
        loss.level_weight(level) * weighted_reconstruction_loss(targets.unsqueeze(1), recon, loss)[0]
        for level, recon in output.reconstructions.items()
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    history = train_model(
        model, loader, loss, TrainConfig(epochs=1, use_lr_schedule=False), optimizer=optimizer
    )
    assert history.loss[0] == pytest.approx(expected.item())


def test_dead_latent_auxiliary_fits_output_residual_and_has_gradients():
    cfg = SedConfig(
        n_neurons=1, n_output_neurons=2, seq_len=1, dsed_topk_map={3: 1},
        decoder=DecoderConfig(output_activation="none"), inference_sparsity="sample_topk",
    )
    model = build_sed(cfg)
    with torch.no_grad():
        model.encoder.projection.weight.zero_()
        model.encoder.projection.bias.copy_(torch.tensor([3.0, 1.0, 0.5]))
        model.decoder.weight.copy_(torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]))
        model.decoder.bias.zero_()
    loader = DataLoader(
        SpikeWindowDataset(torch.zeros(3, 1), 1, targets=torch.tensor([[4.0, 2.0]]).repeat(3, 1)),
        batch_size=1,
    )
    encoder_grads, decoder_grads = [], []
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    original_step = optimizer.step

    def record_step(*args, **kwargs):
        encoder_grads.append(model.encoder.projection.bias.grad.clone())
        decoder_grads.append(model.decoder.weight.grad.clone())
        return original_step(*args, **kwargs)

    optimizer.step = record_step
    history = train_model(
        model, loader, LossConfig(type="mse", timebin_weights=[1]),
        TrainConfig(epochs=1, use_lr_schedule=False, dead_feature_window=1, log_frequency=1),
        optimizer=optimizer,
    )
    assert encoder_grads[0][1].item() == 0
    assert encoder_grads[-1][1].item() == pytest.approx(-1)
    assert decoder_grads[-1][0, 0, 1].item() == pytest.approx(-1)
    assert history.loss[2] == pytest.approx(3.5)  # Main 2.5 + target residual auxiliary 1.


def test_threshold_calibration_depends_on_inputs_with_distinct_output_width():
    cfg = SedConfig(n_neurons=3, n_output_neurons=2, seq_len=1, dsed_topk_map={4: 2})
    inputs, targets = torch.arange(24).reshape(8, 3).float(), torch.full((8, 2), 100.0)
    loader = DataLoader(SpikeWindowDataset(inputs, 1, targets=targets), batch_size=3)
    model = build_sed(cfg)
    train_model(
        model, loader, LossConfig(type="mse", timebin_weights=[1]), TrainConfig(epochs=1)
    )
    cutoffs = []
    with torch.no_grad():
        for batch in loader:
            acts = torch.relu(model.encoder(batch.values))
            cutoffs.append(acts.flatten().topk(len(batch.values) * 2).values[-1])
    torch.testing.assert_close(model.sparsifier.inference_threshold(4), torch.stack(cutoffs).mean())


def test_optional_output_width_defaults_and_preserves_positional_api():
    cfg = SedConfig(3, 1, {4: 1}, EncoderConfig(), DecoderConfig())
    assert cfg.n_output_neurons == 3
    with pytest.raises(ValueError, match="n_output_neurons"):
        SedConfig(3, 1, {4: 1}, n_output_neurons=0)
