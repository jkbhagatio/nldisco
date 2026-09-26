"""Canonical Window names preserve older configuration and checkpoint contracts."""

from dataclasses import asdict

import pytest
import torch

from nldisco import EncoderConfig, SedConfig, build_sed
from nldisco.model import FlatWindowEncoder, TransformerWindowEncoder
from nldisco.model.encoder import TemporalTransformerEncoder


@pytest.mark.parametrize("name, expected", [
    ("FlatWindow", "FlatWindow"), ("flat", "FlatWindow"), ("flat_window", "FlatWindow"),
    ("TransformerWindow", "TransformerWindow"), ("temporal_transformer", "TransformerWindow"),
])
def test_config_serializes_canonical_name(name, expected):
    assert asdict(EncoderConfig(type=name))["type"] == expected


def test_window_is_a_family_not_an_ambiguous_encoder():
    with pytest.raises(ValueError, match="Unknown encoder type"):
        EncoderConfig(type="Window")


def test_factory_accepts_legacy_config_restored_without_post_init():
    config = SedConfig(n_neurons=3, seq_len=4, dsed_topk_map={8: 2})
    # Pickled dataclasses may restore old values without calling __post_init__.
    config.encoder.type = "flat"
    assert isinstance(build_sed(config).encoder, FlatWindowEncoder)


@pytest.mark.parametrize("legacy, current, encoder_class", [
    ("flat", "FlatWindow", FlatWindowEncoder),
    ("temporal_transformer", "TransformerWindow", TransformerWindowEncoder),
])
def test_legacy_state_dict_loads_and_outputs_are_identical(legacy, current, encoder_class):
    def make(name):
        return build_sed(SedConfig(
            n_neurons=3, seq_len=4, dsed_topk_map={8: 2},
            encoder=EncoderConfig(type=name, d_model=8, n_heads=2, n_layers=1),
        )).eval()

    old, new = make(legacy), make(current)
    new.load_state_dict(old.state_dict(), strict=True)
    assert isinstance(new.encoder, encoder_class)
    x = torch.randn(2, 4, 3)
    with torch.no_grad():
        torch.testing.assert_close(old.encoder(x), new.encoder(x), rtol=0, atol=0)
    assert TemporalTransformerEncoder is TransformerWindowEncoder
