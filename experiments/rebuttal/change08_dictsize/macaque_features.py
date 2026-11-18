"""The paper's four macaque feature definitions (current v4 cutoffs) on the development windows."""
from pathlib import Path

import numpy as np

from experiments.churchland.sweeps.s10_endpoint_v1 import comparison_features as shared
from experiments.churchland.sweeps.s10_endpoint_v1 import feature_decoding as decoder
from experiments.churchland.sweeps.s10_endpoint_v1 import nice_cutoffs_revision as revision

MAIN_CHECKOUT = Path('/nfs/nhome/live/jbhagat/nldisco')  # git-ignored data live here (read-only)
SWEEP = shared.SWEEP
FEATURES = shared.EXPECTED_FEATURES
NAMES = dict(recent_braking='Recent deceleration', early_leftward='Early leftward movement',
             fast_target_specific='Fast leftward reach (target)', late_target_specific='Late reach (target)')


def load_features() -> dict:
    """Return name -> FeatureData (rows into the 130,507 export windows, labels, trials)."""
    shared.ROOT = decoder.ROOT = MAIN_CHECKOUT
    decoder.DATA = MAIN_CHECKOUT / 'experiments/churchland/data/nitschke_20090812'
    return revision.features()


def activation_path(folder: Path, dimension: int) -> Path:
    """Largest-level export of one macaque run."""
    return folder / f'activations_level_{dimension}.npy'


def existing_runs() -> list:
    """Saved macaque NLDisco runs with k=16 and two levels: (D, seed, folder, role)."""
    width = SWEEP / 'width192_checks'
    runs = [(128, 0, SWEEP / 'nldisco/d128_k16_l2_seed0', 'existing'),
            (256, 0, SWEEP / 'nldisco/d256_k16_l2_seed0', 'existing'),
            (192, 0, width / 'nldisco/d192_k16_l2_seed0', 'existing')]
    runs += [(192, s, width / f'nldisco_seeds_1_to_5/d192_k16_l2_seed{s}', 'paper' if s == 1 else 'existing')
             for s in range(1, 6)]
    return runs
