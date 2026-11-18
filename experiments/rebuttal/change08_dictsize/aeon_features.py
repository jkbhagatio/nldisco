"""The paper's four Aeon feature definitions on all exported windows."""
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.aeon.sweep_20260921.paper_results import definitions
from experiments.rebuttal.change08_dictsize.dictsize import run_groups

SWEEP = Path('/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20')
FEATURES = ('wheel', 'area', 'speed', 'direction')
#: paper's chosen (D, latent) for each feature
PAPER = dict(wheel=(192, 52), area=(256, 107), speed=(256, 109), direction=(256, 10))
NAMES = dict(wheel='Wheel activity', area='Smaller area near patch', speed='Fast centroid speed',
             direction='Fast outward movement')


def load_features(sweep: Path = SWEEP) -> tuple:
    """Return (endpoints, {name: (rows, condition, groups)}) with rows indexing exported windows."""
    behavior = pd.read_parquet(sweep / 'data/behavior.parquet')
    endpoints = np.load(sweep / 'd192/k12_seed0/endpoint_bins.npy')
    conditions, _ = definitions(behavior, endpoints)
    out = {}
    for name, (condition, valid) in conditions.items():
        rows = np.flatnonzero(valid)
        out[name] = (rows, condition[rows].astype(bool), run_groups(condition[rows].astype(bool), endpoints[rows]))
    return endpoints, out
