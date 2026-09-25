"""Fast neural-only checks for endpoint indexing and published dynamics."""
import io
import pytest

pytest.importorskip("pytorch_lightning", reason="Use the documented LangevinFlow overlay for this test")

import numpy as np
import torch

from experiments.churchland.sweeps.s10_endpoint_v1.langevinflow import extract, make_model


def test_endpoint_and_checkpoint():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    config = dict(hidden_size=32, learning_rate=.001)
    model = make_model(4, config, "cpu").eval()
    model.posterior_mean = True
    values = torch.randn(2, 10, 4)
    seen = []
    hook = model.encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    _, latent, _ = model(values)
    hook.remove()
    assert len(seen) == 10
    for j in range(10):
        torch.testing.assert_close(seen[j], values[:, j])
    changed = values.clone()
    changed[:, 9] += 2
    _, other, _ = model(changed)
    assert not torch.allclose(latent[:, 9, 64:], other[:, 9, 64:])
    # The pinned potential ignores observations: only h responds at the final bin.
    torch.testing.assert_close(latent[:, 9, :64], other[:, 9, :64])
    for j in range(1, 10):
        q, p = latent[:, j-1, :32], latent[:, j-1, 32:64]
        force = torch.autograd.grad(model.potential(q, values[:, j]).sum(), q,
                                    retain_graph=True)[0]
        torch.testing.assert_close(latent[:, j, :32], q + .01 * p)
        torch.testing.assert_close(latent[:, j, 32:64], .45 * (p - .01 * force))
    counts = torch.randn(23, 4)
    indices = np.array([np.arange(3, 13)])
    endpoint = extract(model, counts, indices, 1)
    outside = counts.clone()
    outside[:3] += 10
    outside[13:] -= 10
    np.testing.assert_array_equal(endpoint, extract(model, outside, indices, 1))
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = make_model(4, config, "cpu")
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    np.testing.assert_array_equal(endpoint, extract(restored, counts, indices, 1))


if __name__ == "__main__":
    test_endpoint_and_checkpoint()
    print("LangevinFlow endpoint, dynamics, isolation, and reload checks passed")
