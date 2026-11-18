"""Fast neural-only check of the 20-bin LangevinFlow adapter (run under the overlay)."""
import numpy as np
import pytest
import torch

from experiments.rebuttal.change03_discovery import data


def test_langevinflow_window_model_consumes_each_bin_once():
    pytest.importorskip("pytorch_lightning",
                        reason="Use the documented LangevinFlow overlay for this test")
    from experiments.rebuttal.change03_discovery import run_langevinflow as lf
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = lf.make_model(4, .001, "cpu", hidden=8).eval()
    model.posterior_mean = True
    values = torch.rand(2, data.SEQ_LEN, 4)
    seen = []
    hook = model.encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    prediction, latent, _ = model(values)
    hook.remove()
    assert len(seen) == data.SEQ_LEN and prediction.shape == values.shape
    for j in range(data.SEQ_LEN):
        torch.testing.assert_close(seen[j], values[:, j])
    assert latent.shape == (2, data.SEQ_LEN, 24)
    counts = torch.rand(50, 4)
    out = np.zeros((2, 24), np.float32)
    lf.extract(model, counts, np.array([0, 30]), out)
    with torch.enable_grad():
        expected = model(torch.stack([counts[0:20], counts[30:50]]))[1][:, -1]
    np.testing.assert_allclose(out, expected.detach().numpy(), atol=1e-6)
    assert lf.CONFIGURATIONS["lr0p003_ramp500"] == dict(learning_rate=.003, ramp=500)
