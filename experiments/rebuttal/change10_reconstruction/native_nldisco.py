"""Export NLDisco's own endpoint-bin reconstruction (counts per bin) for every shared window.

Rebuilds the frozen selected 192-wide SED from its saved checkpoint, runs it exactly as the
original export did (z-scored windows, endpoint-only occurrence mask), takes the level-192
reconstruction of the last bin and undoes the z-scoring. Also checks that the recomputed
endpoint activations equal the saved representation. No retraining; writes only to OUT::

    python -m experiments.rebuttal.change10_reconstruction.native_nldisco
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from beartype import beartype
from torch.utils.data import DataLoader

from experiments.churchland.sweeps.s10_endpoint_v1 import nldisco as training
from experiments.rebuttal.change10_reconstruction.reconstruction import DATA, OUT, SWEEP, METHODS, N_WINDOWS
from nldisco.data import SpikeWindowDataset
from nldisco.model import build_sed
from experiments.churchland.scripts.train_nldisco import prepare_inputs


@beartype
def main() -> None:
    """Recompute and save native rates; verify activations against the frozen export."""
    folder = Path(next(m for m in METHODS if m.key == 'nldisco').representation).parent
    config = json.loads((folder / 'config.json').read_text())
    cfg = training.configuration(191, 192, 16, 2)
    with np.load(DATA / 'metadata.npz') as meta:
        trials = meta['trial_id']
    inputs, scale, mean = prepare_inputs(np.load(DATA / 'counts.npy').astype(np.float32), trials, 0.0, 'unit_zscore')
    with np.load(folder / 'preprocessing.npz') as saved:
        np.testing.assert_allclose(saved['scale'], scale, rtol=1e-6)
        np.testing.assert_allclose(saved['mean'], mean, rtol=1e-6, atol=1e-7)
    dataset = SpikeWindowDataset(torch.from_numpy(inputs), seq_len=10, stride=1, trial_ids=trials,
                                 occurrence_support=(9, 0))
    with np.load(SWEEP / 'shared/index.npz') as index:
        anchors = index['anchor_index']
    np.testing.assert_array_equal(np.asarray(dataset.valid_starts) + 9, anchors)
    model = build_sed(cfg).cuda()
    model.load_state_dict(torch.load(folder / 'final_model.pt', weights_only=True)['state_dict'])
    model.eval()
    saved_acts = np.load(folder / 'activations_level_192.npy', mmap_mode='r')
    rates = np.empty((N_WINDOWS, 191), np.float32)
    position, max_act_diff = 0, 0.
    scale_t, mean_t = (torch.as_tensor(a, device='cuda') for a in (scale, mean))
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=1024):
            n = len(batch.values)
            out = model(batch.values.cuda(), occurrence_mask=batch.occurrence_mask.cuda())
            acts = out.sparse_acts[192][:, 9].cpu().numpy()
            max_act_diff = max(max_act_diff, float(np.abs(acts - saved_acts[position:position + n]).max()))
            rates[position:position + n] = (out.reconstructions[192][:, 9] * scale_t + mean_t).cpu().numpy()
            position += n
    assert position == N_WINDOWS and np.isfinite(rates).all()
    if max_act_diff > 1e-4:
        raise AssertionError(f'Recomputed activations differ from the frozen export by {max_act_diff}')
    (OUT / 'native').mkdir(parents=True, exist_ok=True)
    np.save(OUT / 'native/nldisco_endpoint_rates.npy', rates)
    np.savez(OUT / 'native/index.npz', anchor_index=anchors)
    (OUT / 'native/nldisco_endpoint_rates.json').write_text(json.dumps(dict(
        checkpoint=str(folder / 'final_model.pt'), config_id=config['config_id'], level=192,
        output='level-192 reconstruction of window bin 9 (endpoint), z-scoring undone: x * scale + mean',
        max_abs_activation_difference_vs_frozen_export=max_act_diff,
        nonpositive_fraction=float((rates <= 0).mean())), indent=2))
    print(f'max |act - frozen| = {max_act_diff:.2e}; nonpositive rates {(rates <= 0).mean():.4f}')


if __name__ == '__main__':
    main()
