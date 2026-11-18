"""Export ST-NDT's own endpoint-bin rate prediction for every shared window (change 1's model).

Loads the selected checkpoint named in change 1's ``REPRESENTATIONS_READY.json`` and runs it
unmasked in validation mode, exactly as that marker specifies, keeping exp(log-rate) at window
bin 9. The unmasked input contains the bin being reconstructed. The representation recomputed
through change 1's own ``extract`` is checked against the saved export. Needs ST-NDT's
interpreter and change 1's code (``--code-root``: the merged repository root by default, or an
extraction of the commit in the marker)::

    /ceph/aeon/aeon/nldisco/rebuttal_20261118/change01_stndt/stndt-venv/bin/python \
        experiments/rebuttal/change10_reconstruction/native_stndt.py [--code-root DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

MARKER = Path('/ceph/aeon/aeon/nldisco/rebuttal_20261118/change01_stndt/REPRESENTATIONS_READY.json')
COUNTS = Path('/nfs/nhome/live/jbhagat/nldisco/experiments/churchland/data/nitschke_20090812/counts.npy')
SHARED = Path('/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1/shared/index.npz')
OUT = Path('/ceph/aeon/aeon/nldisco/rebuttal_20261118/change10_reconstruction')


def main() -> None:
    """Recompute and save native rates; verify the representation against the frozen export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root))
    from experiments.rebuttal.change01_stndt.stndt import Hyperparameters, extract, make_model

    info = json.loads(MARKER.read_text())
    folder = Path(info['representation']['path']).parent
    checkpoint = torch.load(info['checkpoint']['path'], weights_only=False, map_location='cuda')
    settings = info['selected_settings']
    hp = Hyperparameters(contrast_weight=settings['contrast_weight'], learning_rate=settings['learning_rate'],
                         dropout=settings['dropout'])
    assert (hp.layers, hp.heads, hp.hidden) == (settings['layers'], settings['heads'], settings['hidden'])
    max_spikes = checkpoint['config']['max_spikes']
    model = make_model(191, 10, hp, 'cuda', max_spikes)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with np.load(folder / 'index.npz') as index, np.load(SHARED) as shared:
        indices = index['source_indices']
        np.testing.assert_array_equal(index['anchor_index'], shared['anchor_index'])
        np.testing.assert_array_equal(indices, shared['source_indices'])
    counts = torch.as_tensor(np.load(COUNTS).astype(np.int64), device='cuda')
    saved = np.load(info['representation']['path'], mmap_mode='r')
    check = extract(model, counts, indices[:4096], args.batch_size)
    max_diff = float(np.abs(check - saved[:4096]).max())
    if max_diff > 1e-4:
        raise AssertionError(f'Recomputed representation differs from frozen export by {max_diff}')
    rates = np.empty((len(indices), 191), np.float32)
    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            spikes = counts[torch.as_tensor(indices[start:start + args.batch_size], device='cuda')]
            log_rates = model(spikes, spikes, contrast_src1=None, contrast_src2=None, val_phase=True,
                              return_outputs=False, return_weights=False)[3]
            rates[start:start + len(spikes)] = log_rates[:, 9].exp().cpu().numpy()
    if not np.isfinite(rates).all():
        raise FloatingPointError('Nonfinite ST-NDT rates')
    (OUT / 'native').mkdir(parents=True, exist_ok=True)
    np.save(OUT / 'native/stndt_endpoint_rates.npy', rates)
    (OUT / 'native/stndt_endpoint_rates.json').write_text(json.dumps(dict(
        checkpoint=info['checkpoint']['path'], config_id=info['selected_config_id'], marker_commit=info['commit'],
        output='exp(log-rate)[:, 9] from the unmasked window in validation mode; input contains the target bin',
        max_abs_representation_difference_vs_frozen_export_first_4096=max_diff), indent=2))
    print(f'max |rep - frozen| (first 4096) = {max_diff:.2e}; rates mean {rates.mean():.3f}')


if __name__ == '__main__':
    main()
