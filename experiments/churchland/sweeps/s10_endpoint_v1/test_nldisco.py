"""Endpoint receptive field, alignment, sparsity, and reload contract checks."""

import io

import numpy as np
import pytest
import torch

from experiments.churchland.sweeps.s10_endpoint_v1.nldisco import configuration
from nldisco.data import SpikeWindowDataset
from nldisco.model import build_sed


@pytest.mark.parametrize('dimension', [128, 256])
@pytest.mark.parametrize('budget', [16, 32, 48])
@pytest.mark.parametrize('levels', [2, 3])
def test_nested_budgets(dimension, budget, levels):
    cfg = configuration(3, dimension, budget, levels)
    assert len(cfg.dsed_topk_map) == levels
    assert cfg.temporal_occurrence_support() == (9, 0)
    assert cfg.encoder.n_layers == 1
    assert cfg.encoder.attention_radius == 9
    model = build_sed(cfg)
    output = model(torch.randn(4, 10, 3))
    for d, k in cfg.dsed_topk_map.items():
        assert torch.count_nonzero(output.sparse_acts[d][:, :9]) == 0
        assert torch.count_nonzero(output.sparse_acts[d]) <= 4 * k


def test_endpoint_index_and_complete_receptive_field():
    torch.manual_seed(5)
    dataset = SpikeWindowDataset(torch.randn(27, 3), 10, trial_ids=np.repeat([0, 1], [12, 15]),
                                 occurrence_support=(9, 0))
    assert len(dataset) == 9
    assert [int(dataset[i].anchor_index) for i in range(len(dataset))] == [9, 10, 11, 21, 22, 23, 24, 25, 26]
    for item in dataset:
        assert item.occurrence_mask.sum() == 1
        assert item.occurrence_mask[9]
        assert item.anchor_index == item.source_indices[9]
    model = build_sed(configuration(3, 128, 16, 2))
    values = torch.randn(2, 10, 3, requires_grad=True)
    model.encoder(values)[:, 9].square().sum().backward()
    assert torch.all(values.grad.abs().sum((0, 2)) > 0)
    # A lone endpoint impulse decodes across exactly the full ten-bin window.
    with torch.no_grad():
        model.decoder.weight.zero_()
        model.decoder.weight[0, 0] = torch.arange(1., 11.)
        model.decoder.bias.zero_()
        code = torch.zeros(1, 10, 128)
        code[0, 9, 0] = 1
        torch.testing.assert_close(model.decode(code)[0, :, 0], torch.arange(1., 11.))


def test_final_model_reload():
    model = build_sed(configuration(3, 128, 16, 3))
    values = torch.randn(3, 10, 3)
    model(values)
    model.eval()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = build_sed(model.cfg)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    restored.eval()
    original, reloaded = model(values), restored(values)
    for level in model.cfg.dsed_topk_map:
        torch.testing.assert_close(original.sparse_acts[level], reloaded.sparse_acts[level])
        torch.testing.assert_close(original.reconstructions[level], reloaded.reconstructions[level])
