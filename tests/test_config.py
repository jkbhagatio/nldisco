import pytest

from nldisco.config import DecoderConfig, EncoderConfig, LossConfig, SedConfig


def test_timebin_weights_must_match_sequence_length():
    sed_cfg = SedConfig(n_neurons=3, seq_len=4, dsed_topk_map={8: 2})
    with pytest.raises(ValueError, match="exactly one positive value"):
        LossConfig(timebin_weights=[1, 1, 1]).validate_for(sed_cfg)


def test_every_timebin_must_have_positive_weight():
    sed_cfg = SedConfig(n_neurons=3, seq_len=3, dsed_topk_map={8: 2})
    with pytest.raises(ValueError, match="Every timebin weight"):
        LossConfig(timebin_weights=[1, 0, 1]).validate_for(sed_cfg)


def test_timebin_weights_must_be_finite():
    sed_cfg = SedConfig(n_neurons=3, seq_len=2, dsed_topk_map={8: 2})
    with pytest.raises(ValueError, match="finite and positive"):
        LossConfig(timebin_weights=[1, float("nan")]).validate_for(sed_cfg)


def test_shift_equivariance_requires_equal_timebin_weights():
    sed_cfg = SedConfig(
        n_neurons=3,
        seq_len=4,
        dsed_topk_map={8: 2},
        encoder=EncoderConfig(
            type="TransformerWindow",
            shift_equivariant=True,
            d_model=8,
            n_heads=2,
            n_layers=1,
            attention_radius=1,
        ),
        decoder=DecoderConfig(temporal_kernel_len=3),
    )
    with pytest.raises(ValueError, match="equal timebin_weights"):
        LossConfig(timebin_weights=[1, 1, 1, 2]).validate_for(sed_cfg)
    LossConfig(timebin_weights=[1, 1, 1, 1]).validate_for(sed_cfg)


def test_flat_encoder_cannot_enable_shift_equivariance():
    with pytest.raises(ValueError, match="only supported"):
        EncoderConfig(type="FlatWindow", shift_equivariant=True)


def test_shift_equivariant_support_must_leave_valid_occurrences():
    with pytest.raises(ValueError, match="leave no valid occurrence"):
        SedConfig(
            n_neurons=3,
            seq_len=4,
            dsed_topk_map={8: 2},
            encoder=EncoderConfig(
                type="TransformerWindow",
                shift_equivariant=True,
                n_layers=2,
                attention_radius=2,
            ),
            decoder=DecoderConfig(temporal_kernel_len=3),
        )


def test_shift_equivariant_topk_cannot_exceed_usable_occurrences():
    with pytest.raises(ValueError, match=r"usable_positions \* d_sed"):
        SedConfig(
            n_neurons=3,
            seq_len=6,
            dsed_topk_map={1: 5},
            encoder=EncoderConfig(
                type="TransformerWindow",
                shift_equivariant=True,
                n_layers=1,
                attention_radius=2,
            ),
            decoder=DecoderConfig(temporal_kernel_len=3),
        )


def test_shift_equivariant_decoder_must_reach_every_edge():
    with pytest.raises(ValueError, match="decoder support must cover encoder context"):
        SedConfig(
            n_neurons=3,
            seq_len=10,
            dsed_topk_map={8: 2},
            encoder=EncoderConfig(
                type="TransformerWindow",
                shift_equivariant=True,
                n_layers=2,
                attention_radius=2,
            ),
            decoder=DecoderConfig(temporal_kernel_len=3),
        )


def test_shift_equivariant_config_reports_usable_occurrence_support():
    cfg = SedConfig(
        n_neurons=3,
        seq_len=8,
        dsed_topk_map={8: 2},
        encoder=EncoderConfig(
            type="TransformerWindow",
            shift_equivariant=True,
            n_layers=1,
            attention_radius=2,
        ),
        decoder=DecoderConfig(temporal_kernel_len=3),
    )
    assert cfg.temporal_occurrence_support() == (2, 0)
    assert cfg.usable_occurrence_positions == 6


@pytest.mark.parametrize("decay", [float("nan"), -0.1, 1.0])
def test_invalid_inference_threshold_decay_is_rejected(decay):
    with pytest.raises(ValueError, match="threshold_decay"):
        SedConfig(
            n_neurons=3,
            seq_len=1,
            dsed_topk_map={8: 2},
            threshold_decay=decay,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_optimization_config_is_rejected(value):
    sed_cfg = SedConfig(n_neurons=3, seq_len=1, dsed_topk_map={8: 2})
    with pytest.raises(ValueError, match="tau must be finite"):
        LossConfig(timebin_weights=[1], tau=value).validate_for(sed_cfg)
    with pytest.raises(ValueError, match="learning_rate must be finite"):
        from nldisco.config import TrainConfig

        TrainConfig(learning_rate=value)


def test_at_least_one_sed_level_must_contribute_to_objective():
    sed_cfg = SedConfig(n_neurons=3, seq_len=1, dsed_topk_map={8: 2, 16: 3})
    with pytest.raises(ValueError, match="At least one SED level"):
        LossConfig(
            timebin_weights=[1],
            dsed_loss_weight_map={8: 0, 16: 0},
        ).validate_for(sed_cfg)
