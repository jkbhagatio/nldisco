"""Fast tests of CEBRA endpoint semantics and official encoder reload."""
import io

import numpy as np
import torch

from experiments.churchland.sweeps.s10_endpoint_v1.cebra import (
    configurations, encoder, pair_endpoints, transform, windows,
)


def test_grid():
    assert len(configurations()) == 12
    assert len({c['config_id'] for c in configurations()}) == 12


def test_pairs_and_windows():
    trials = np.repeat(np.arange(3), [20, 11, 9])
    endpoints = np.r_[np.arange(9, 20), np.arange(29, 31)].astype(np.int64)
    assert np.array_equal(pair_endpoints(endpoints, trials, 5), np.arange(9, 15))
    assert np.array_equal(pair_endpoints(endpoints, trials, 1), np.r_[np.arange(9, 19), 29])
    data = torch.arange(80, dtype=torch.float32).reshape(40, 2)
    x = windows(data, torch.tensor([9, 29]))
    assert x.shape == (2, 2, 10)
    torch.testing.assert_close(x[0].T, data[:10])
    torch.testing.assert_close(x[1].T, data[20:30])


def test_encoder_and_reload():
    torch.manual_seed(0)
    model = encoder(4, 32, 'cpu')
    data = torch.randn(24, 4)
    endpoints = np.arange(9, 24, dtype=np.int64)
    assert model(windows(data, torch.tensor([9, 10]))).shape == (2, 32)
    output = transform(model, data, endpoints, 7)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    loaded = encoder(4, 32, 'cpu')
    loaded.load_state_dict(torch.load(buffer, weights_only=True))
    np.testing.assert_allclose(output, transform(loaded, data, endpoints, 7), atol=1e-7)
    perturbed = data.clone()
    perturbed[10:] += 100
    np.testing.assert_allclose(transform(model, data, endpoints[:1]),
                               transform(model, perturbed, endpoints[:1]), atol=1e-7)
