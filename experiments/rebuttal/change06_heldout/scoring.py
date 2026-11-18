"""Select latents on the training part and score them on the held-out part.

``evaluate_feature`` is the single entry point used for both datasets: it receives
activations, labels, a part indicator (0 training, 1 held-out) and grouping ids
(trials or episodes); the selection step is given the training rows only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from beartype import beartype

from experiments.rebuttal.change06_heldout import metrics

CRITERIA = ("auroc", "sel")


@beartype
def evaluate_feature(values: np.ndarray, labels: np.ndarray, part: np.ndarray, groups: np.ndarray,
                     operating_fpr: Optional[float] = None) -> dict:
    """Pick the best-matching latent on the training part and score both parts.

    Returns, per selection criterion, the selected latent with its training-part and
    held-out scores (native activity ``z > 0``) and, if ``operating_fpr`` is given, the
    scores at a threshold fitted on the training part to that false-positive rate.
    A diagnostic ``oracle_heldout_auroc`` (best eligible held-out AUROC over all latents)
    is reported but never used for selection.
    """
    train, held = part == 0, part == 1
    table = metrics.score_latents(values[train], labels[train], groups[train])
    heldout_table = metrics.score_latents(values[held], labels[held], groups[held])
    out = dict(oracle_heldout_auroc=float(np.nanmax(heldout_table["auroc"]))
               if heldout_table["eligible"].any() else float("nan"))
    for criterion in CRITERIA:
        latent = metrics.select_latent(table, criterion)
        z_train, z_held = values[train, latent], values[held, latent]
        record = dict(latent=latent,
                      train=metrics.score_one(z_train, labels[train], 0.0, groups[train]),
                      heldout=metrics.score_one(z_held, labels[held], 0.0, groups[held]))
        if operating_fpr is not None:
            threshold = metrics.operating_threshold(z_train, labels[train], operating_fpr)
            strict = float(np.nextafter(np.float32(threshold), np.float32(-np.inf)))
            record["operating_threshold"] = threshold
            record["train_operating"] = metrics.score_one(z_train, labels[train], strict, groups[train])
            record["heldout_operating"] = metrics.score_one(z_held, labels[held], strict, groups[held])
        out[criterion] = record
    return out
