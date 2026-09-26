import pytest
import torch as t

from experiments.attribution import encoder_attribution, mean_encoder_attribution
from nldisco.config import DecoderConfig, EncoderConfig, SedConfig
from nldisco.model import build_sed


def _transformer(shift_equivariant: bool) -> SedConfig:
    return SedConfig(
        n_neurons=7,
        seq_len=5,
        dsed_topk_map={12: 3},
        encoder=EncoderConfig(
            type="TransformerWindow",
            shift_equivariant=shift_equivariant,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_feedforward=16,
            causal=True,
            attention_radius=4,
        ),
        decoder=DecoderConfig(temporal_kernel_len=5, output_activation="none"),
    )


def test_linear_encoder_attribution_equals_weight_times_input():
    t.manual_seed(0)
    model = build_sed(SedConfig(n_neurons=6, seq_len=3, dsed_topk_map={10: 2})).eval()
    x = t.randn(4, 3, 6)
    weight = model.encoder.projection.weight[5].reshape(3, 6)
    t.testing.assert_close(encoder_attribution(model, x, 5), x * weight)


@pytest.mark.parametrize("shift_equivariant", [False, True])
def test_transformer_attribution_matches_finite_differences(shift_equivariant):
    t.manual_seed(0)
    model = build_sed(_transformer(shift_equivariant)).double().eval()
    x = t.randn(2, 5, 7, dtype=t.float64)
    attribution = encoder_attribution(model, x, 4)

    def endpoint(inputs):
        pre = model.encoder(inputs)
        return pre[:, -1, 4] if pre.ndim == 3 else pre[:, 4]

    eps = 1e-6
    for b, s, i in [(0, 4, 2), (1, 0, 6), (1, 3, 1)]:
        bumped = x.clone()
        bumped[b, s, i] += eps
        derivative = (endpoint(bumped)[b] - endpoint(x)[b]) / eps
        assert attribution[b, s, i].item() == pytest.approx(
            (x[b, s, i] * derivative).item(), abs=1e-6
        )


def test_attribution_rejects_out_of_range_latent():
    model = build_sed(SedConfig(n_neurons=3, seq_len=1, dsed_topk_map={4: 1}))
    with pytest.raises(ValueError, match="latent"):
        encoder_attribution(model, t.zeros(1, 1, 3), 4)


def test_mean_attribution_streams_over_batches():
    t.manual_seed(1)
    model = build_sed(_transformer(True)).eval()
    x = t.randn(10, 5, 7)
    expected = encoder_attribution(model, x, 1).mean(dim=0).numpy()
    streamed = mean_encoder_attribution(model, [x[:3], x[3:7], x[7:]], 1)
    assert streamed.shape == (5, 7)
    assert abs(streamed - expected).max() < 1e-5
    with pytest.raises(ValueError, match="No windows"):
        mean_encoder_attribution(model, [], 1)
