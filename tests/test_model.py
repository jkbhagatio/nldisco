import pytest
import torch as t

from nldisco.config import DecoderConfig, EncoderConfig, SedConfig
from nldisco.model import build_sed
from nldisco.model.decoder import TemporalConvDecoder
from nldisco.model.sparsify import BatchTopK, batch_topk


def _sed_config(encoder: EncoderConfig) -> SedConfig:
    return SedConfig(
        n_neurons=5,
        seq_len=6,
        dsed_topk_map={8: 2},
        encoder=encoder,
        decoder=DecoderConfig(temporal_kernel_len=3),
    )


def test_flat_window_and_transformer_window_reconstruct_full_windows():
    inputs = t.rand(4, 6, 5)
    configs = [
        EncoderConfig(type="FlatWindow"),
        EncoderConfig(
            type="TransformerWindow",
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_feedforward=32,
            attention_radius=1,
        ),
    ]
    for encoder_cfg in configs:
        output = build_sed(_sed_config(encoder_cfg))(inputs)
        assert output.reconstructions[8].shape == (4, 6, 5)
        assert output.sparse_acts[8].shape == (4, 8)


def test_shift_equivariant_transformer_has_temporal_sparse_code():
    cfg = _sed_config(
        EncoderConfig(
            type="TransformerWindow",
            shift_equivariant=True,
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_feedforward=32,
        )
    )
    output = build_sed(cfg)(t.rand(4, 6, 5))
    assert output.code_layout == "temporal"
    assert output.sparse_acts[8].shape == (4, 6, 8)
    assert output.reconstructions[8].shape == (4, 6, 5)
    assert (output.sparse_acts[8] > 0).sum() <= 4 * 2
    assert not output.sparse_acts[8][:, :2].any()


def test_batch_topk_does_not_modify_dense_activations():
    acts = t.arange(24, dtype=t.float32).reshape(3, 8)
    original = acts.clone()
    sparse = batch_topk(acts, average_top_k=2)
    assert t.equal(acts, original)
    assert (sparse > 0).sum() == 6


def test_calibrated_inference_threshold_preserves_batch_topk_semantics():
    scores = t.tensor([[10.0, 9.0, 8.0], [1.0, 0.5, 0.1]])
    sparsifier = BatchTopK({3: 1}, threshold_decay=0)
    training = sparsifier(scores)[3]
    sparsifier.eval()
    evaluation = sparsifier(scores)[3]
    assert (training > 0).sum(dim=1).tolist() == [2, 0]
    assert (evaluation > 0).sum(dim=1).tolist() == [2, 0]
    assert t.equal(training, evaluation)


def test_uncalibrated_threshold_requires_explicit_sample_topk_policy():
    scores = t.rand(2, 3)
    threshold_sparsifier = BatchTopK({3: 1}).eval()
    with pytest.raises(RuntimeError, match="uncalibrated"):
        threshold_sparsifier(scores)
    sample_sparsifier = BatchTopK({3: 1}, inference_sparsity="sample_topk").eval()
    assert (sample_sparsifier(scores)[3] > 0).sum(dim=1).tolist() == [1, 1]


def test_temporal_decoder_is_shift_equivariant_in_valid_interior():
    decoder = TemporalConvDecoder(
        n_features=2,
        n_neurons=3,
        kernel_len=3,
        alignment="causal",
        output_activation="none",
    )
    code = t.zeros(1, 8, 2)
    code[:, 4, 0] = 1
    shifted_code = t.zeros_like(code)
    shifted_code[:, 5, 0] = 1
    reconstruction = decoder(code, include_bias=False)
    shifted_reconstruction = decoder(shifted_code, include_bias=False)
    assert t.allclose(shifted_reconstruction[:, 1:], reconstruction[:, :-1])


def test_relative_bounded_transformer_encoder_is_shift_equivariant_in_interior():
    cfg = SedConfig(
        n_neurons=3,
        seq_len=10,
        dsed_topk_map={6: 2},
        encoder=EncoderConfig(
            type="TransformerWindow",
            shift_equivariant=True,
            d_model=12,
            n_heads=3,
            n_layers=2,
            d_feedforward=24,
            dropout=0,
            causal=True,
            attention_radius=1,
        ),
        decoder=DecoderConfig(temporal_kernel_len=3, temporal_alignment="causal"),
    )
    model = build_sed(cfg).eval()
    inputs = t.zeros(1, 10, 3)
    inputs[:, 3:7] = t.randn(1, 4, 3)
    shifted = t.zeros_like(inputs)
    shifted[:, 4:8] = inputs[:, 3:7]
    with t.no_grad():
        original_scores = model.encoder(inputs)
        shifted_scores = model.encoder(shifted)
    assert t.allclose(original_scores[:, 2:9], shifted_scores[:, 3:10], atol=1e-6)


def test_padding_masks_occurrence_and_kernel_support():
    cfg = _sed_config(
        EncoderConfig(
            type="TransformerWindow",
            shift_equivariant=True,
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_feedforward=32,
            attention_radius=1,
        )
    )
    model = build_sed(cfg)
    padding_mask = t.tensor([[False, False, False, False, True, True]])
    output = model(t.rand(1, 6, 5), padding_mask=padding_mask)
    assert not output.sparse_acts[8][:, 4:].any()
