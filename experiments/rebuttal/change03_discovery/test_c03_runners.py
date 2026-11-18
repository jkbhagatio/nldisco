"""Fast checks of window alignment and export shape for the Aeon-window runners."""
import numpy as np
import pytest
import torch

from experiments.rebuttal.change03_discovery import data, run_pca


def test_pca_windows_are_causal_and_flat():
    values = torch.arange(60, dtype=torch.float32).reshape(30, 2)
    flat = run_pca.flat_windows(values, torch.tensor([0, 10]))
    assert flat.shape == (2, 40)
    torch.testing.assert_close(flat[1], values[10:30].reshape(-1))


def test_cebra_encoder_sees_exactly_twenty_bins():
    pytest.importorskip("cebra")
    from experiments.rebuttal.change03_discovery import run_cebra
    torch.manual_seed(0)
    model = run_cebra.encoder(5, 192).eval()
    values = torch.randn(64, 5)
    starts = torch.tensor([0, 7, 40])
    x = run_cebra.windows(values, starts)
    assert x.shape == (3, 5, 20)
    torch.testing.assert_close(x[1].T, values[7:27])
    with torch.inference_mode():
        out = model(x)
        assert out.shape == (3, 192)
        torch.testing.assert_close(out.norm(dim=1), torch.ones(3), atol=1e-5, rtol=0)
        with pytest.raises(Exception):
            assert model(x[:, :, 1:]).shape == (3, 192)
        for position in (0, 19):  # the first and last bin both reach the output
            changed = x.clone()
            changed[:, :, position] += 1.
            assert not torch.allclose(model(changed), out)
    out_memmap = np.zeros((3, 192), np.float32)
    run_cebra.transform(model, values, starts.numpy(), out_memmap)
    np.testing.assert_allclose(out_memmap, out.numpy(), atol=1e-6)

