"""Small checks for immutable CEBRA export validation."""
import numpy as np
import pytest

from experiments.churchland.sweeps.s10_endpoint_v1.cebra_feature_selection import (
    FEATURES,
    choose_configuration,
    load_export,
    sha256,
)


def test_export_validation(tmp_path):
    path = tmp_path / 'export.npz'
    source = np.array([9, 10, 29])
    values = np.eye(3, dtype=np.float32)
    np.savez(path, source_index=source, activations=values)
    np.testing.assert_array_equal(load_export(path, source, 3), values)
    assert len(sha256(path)) == 64
    with pytest.raises(AssertionError):
        load_export(path, source[::-1], 3)
    with pytest.raises(ValueError):
        load_export(path, source, 4)


def test_four_feature_single_configuration_gate():
    choices_a = {name: {'auroc': value} for name, value in zip(FEATURES, (.99, .6, .6, .6))}
    choices_b = {name: {'auroc': .75} for name in FEATURES}
    configs = [dict(config_id='a', feature_choices=choices_a), dict(config_id='b', feature_choices=choices_b)]
    assert choose_configuration(configs, FEATURES)['config_id'] == 'b'
    with pytest.raises(ValueError):
        choose_configuration(configs, FEATURES[:3])
    incomplete = [dict(config_id='missing', feature_choices={name: None for name in FEATURES})]
    with pytest.raises(ValueError):
        choose_configuration(incomplete, FEATURES)
