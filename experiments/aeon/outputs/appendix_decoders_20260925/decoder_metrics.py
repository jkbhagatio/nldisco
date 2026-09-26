"""Decision-threshold selection and scoring for frozen behavioral decoders."""

# ruff: noqa: F821 -- jaxtyping dimension names are runtime shape annotations.

import numpy as np
from beartype import beartype
from jaxtyping import Bool, Float, Integer, jaxtyped


@jaxtyped(typechecker=beartype)
def select_balanced_threshold(
    labels: Bool[np.ndarray, "n"], probabilities: Float[np.ndarray, "n"]
) -> float:
    """Maximize validation balanced accuracy; ties choose the highest threshold.

    Predictions use ``probability >= threshold``. Equal probabilities stay
    together. The all-negative operating point is included. Integer counts
    resolve objective ties exactly, without floating-point rounding effects.
    """
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and within [0, 1].")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Threshold selection requires both label classes.")
    order = np.argsort(-probabilities, kind="stable")
    scores = probabilities[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.r_[0, np.cumsum(labels[order], dtype=np.int64)[ends]]
    fp = np.r_[0, ends + 1 - tp[1:]]
    thresholds = np.r_[np.nextafter(float(scores[0]), np.inf), scores[ends]]
    return float(thresholds[np.argmax(tp * negatives - fp * positives)])


@jaxtyped(typechecker=beartype)
def binary_metrics(
    labels: Bool[np.ndarray, "n"], predicted: Bool[np.ndarray, "n"]
) -> dict:
    """Return confusion counts, sensitivity, specificity, and balanced accuracy."""
    tp, fn = int(np.sum(labels & predicted)), int(np.sum(labels & ~predicted))
    tn, fp = int(np.sum(~labels & ~predicted)), int(np.sum(~labels & predicted))
    if not tp + fn or not tn + fp:
        raise ValueError("Balanced accuracy requires both label classes.")
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    return dict(tp=tp, fn=fn, tn=tn, fp=fp, sensitivity=sensitivity,
                specificity=specificity, balanced_accuracy=(sensitivity + specificity) / 2)


@jaxtyped(typechecker=beartype)
def trial_balanced_interval(
    labels: Bool[np.ndarray, "n"], predicted: Bool[np.ndarray, "n"],
    trials: Integer[np.ndarray, "n"], repeats: int = 300, seed: int = 734,
) -> dict:
    """Percentile interval from whole-trial resampling with threshold fixed."""
    if repeats < 1:
        raise ValueError("At least one bootstrap replicate is required.")
    binary_metrics(labels, predicted)
    _, inverse = np.unique(trials, return_inverse=True)
    counts = np.stack([np.bincount(inverse, weights=mask) for mask in
                       (labels & predicted, labels & ~predicted,
                        ~labels & ~predicted, ~labels & predicted)], axis=1)
    rng = np.random.default_rng(seed)
    samples = counts[rng.integers(len(counts), size=(repeats, len(counts)))].sum(axis=1)
    tp, fn, tn, fp = samples.T
    valid = ((tp + fn) > 0) & ((tn + fp) > 0)
    if not valid.any():
        raise ValueError("No bootstrap replicate contains both classes.")
    values = .5 * (tp[valid] / (tp + fn)[valid] + tn[valid] / (tn + fp)[valid])
    low, high = np.quantile(values, [.025, .975])
    return dict(ci_lower=float(low), ci_upper=float(high),
                bootstrap_valid_replicates=int(valid.sum()))
