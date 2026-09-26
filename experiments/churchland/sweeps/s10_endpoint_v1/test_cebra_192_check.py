"""Tests for the supplementary width-only CEBRA check."""
import numpy as np
import pytest
import torch

from experiments.churchland.sweeps.s10_endpoint_v1 import cebra
from experiments.churchland.sweeps.s10_endpoint_v1.cebra_192_check import choose


def test_selection_eligibility_and_tie():
    rows = [dict(latent_id=4, eligible=True, sel=.9, auroc=.8),
            dict(latent_id=3, eligible=True, sel=.9, auroc=.7),
            dict(latent_id=1, eligible=False, sel=1., auroc=1.)]
    assert choose(rows, 'sel')['latent_id'] == 3
    assert choose(rows, 'auroc')['latent_id'] == 4
    with pytest.raises(ValueError):
        choose(rows, 'coverage')


def test_width192_output_isolation():
    torch.manual_seed(0)
    model = cebra.encoder(4, 192, 'cpu')
    data = torch.randn(24, 4)
    anchors = np.array([9, 20], dtype=np.int64)
    first = cebra.transform(model, data, anchors)
    assert first.shape == (2, 192)
    modified = data.clone()
    modified[10:] += 20
    np.testing.assert_array_equal(cebra.transform(model, data, anchors[:1]),
                                  cebra.transform(model, modified, anchors[:1]))
