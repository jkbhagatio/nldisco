"""Export LangevinFlow's own endpoint-bin rate prediction for every shared window.

Runs the frozen selected checkpoint deterministically (eval mode, posterior means, zero
innovations, as in the original representation export) and keeps exp(log-rate) at window
bin 9. Also checks the recomputed endpoint latents against the frozen representation.
Needs the LangevinFlow environment; run from the repository root::

    /nfs/nhome/live/jbhagat/nldisco/experiments/churchland/environments/langevinflow-venv/bin/python \
        -m experiments.rebuttal.change10_reconstruction.native_langevinflow
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

MAIN = Path('/nfs/nhome/live/jbhagat/nldisco')
sys.path.insert(0, str(MAIN / 'experiments/churchland/environments/LangevinFlow_CCN'))
from experiments.churchland.sweeps.s10_endpoint_v1.langevinflow import make_model  # noqa: E402

SWEEP = Path('/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1')
RUN = SWEEP / 'langevinflow/h64_lr0p003_ramp500'
DATA = MAIN / 'experiments/churchland/data/nitschke_20090812'
OUT = Path('/ceph/aeon/aeon/nldisco/rebuttal_20261118/change10_reconstruction')


def main() -> None:
    """Recompute and save native rates; verify latents against the frozen export."""
    torch.manual_seed(0)
    checkpoint = torch.load(RUN / 'model_final.pt', weights_only=False, map_location='cuda')
    config = checkpoint['config']
    counts = torch.as_tensor(np.load(DATA / 'counts.npy').astype(np.float32), device='cuda')
    with np.load(RUN / 'index.npz') as index, np.load(SWEEP / 'shared/index.npz') as shared:
        indices = index['source_indices']
        np.testing.assert_array_equal(index['anchor_index'], shared['anchor_index'])
    model = make_model(counts.shape[1], config, 'cuda')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.posterior_mean = True
    saved = np.load(RUN / 'activations.npy', mmap_mode='r')
    rates = np.empty((len(indices), counts.shape[1]), np.float32)
    max_diff = 0.
    for start in range(0, len(indices), 1024):
        values = counts[torch.as_tensor(indices[start:start + 1024], device='cuda')]
        with torch.enable_grad():  # the potential's force needs autograd, as in the original export
            log_rates, latent, _ = model(values)
        latent = latent[:, 9].detach().cpu().numpy()
        max_diff = max(max_diff, float(np.abs(latent - saved[start:start + len(values)]).max()))
        rates[start:start + len(values)] = log_rates[:, 9].detach().exp().cpu().numpy()
    if max_diff > 1e-4 or not np.isfinite(rates).all():
        raise AssertionError(f'Recomputed latents differ from frozen export by {max_diff}')
    (OUT / 'native').mkdir(parents=True, exist_ok=True)
    np.save(OUT / 'native/langevinflow_endpoint_rates.npy', rates)
    (OUT / 'native/langevinflow_endpoint_rates.json').write_text(json.dumps(dict(
        checkpoint=str(RUN / 'model_final.pt'), config_id=config['config_id'],
        output='exp(readout(decoder(latents)))[:, 9]: endpoint-bin Poisson rate, eval mode, posterior means',
        max_abs_latent_difference_vs_frozen_export=max_diff), indent=2))
    print(f'max |latent - frozen| = {max_diff:.2e}')


if __name__ == '__main__':
    main()
