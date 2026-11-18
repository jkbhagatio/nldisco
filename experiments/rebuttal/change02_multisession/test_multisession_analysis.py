"""Fast checks of the multi-session scoring, loading and transfer-metric helpers."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from experiments.aeon.sweep_20260921.shared_analysis import Condition
from experiments.aeon.sweep_20260921.shared_training import WindowBatches
from experiments.rebuttal.change02_multisession.adapt import GpuWindowBatches, score_run
from experiments.rebuttal.change02_multisession.common import score_latents, score_one
from experiments.rebuttal.change02_multisession.new_feature import LAGS, peri_event
from experiments.rebuttal.change02_multisession.profiles import (
    condition_transfer, identity_transfer, tuning_profile,
)


def _sparse_codes(n: int = 4000, d: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    condition = rng.random(n) < .2
    codes = np.where(rng.random((n, d)) < .3, rng.gamma(2., 1., (n, d)), 0.).astype(np.float32)
    codes[:, 2] = np.where(condition & (rng.random(n) < .9), rng.gamma(4., 1., n), codes[:, 2] * (rng.random(n) < .2))
    valid = rng.random(n) < .9
    return codes, condition, valid


def test_score_latents_matches_reference_scoring_with_tied_zeros():
    codes, condition, valid = _sparse_codes()
    table = score_latents(codes, condition, valid)
    for latent in range(codes.shape[1]):
        reference = score_one(codes[:, latent].astype(np.float64), condition, valid)
        row = table.iloc[latent]
        np.testing.assert_allclose(row.auroc, roc_auc_score(condition[valid], codes[valid, latent]), atol=1e-9)
        np.testing.assert_allclose([row.coverage, row.specificity, row.selectivity, row.auroc],
                                   [reference[key] for key in ("coverage", "specificity", "selectivity", "auroc")],
                                   atol=1e-9)
    assert int(table.auroc.idxmax()) == 2


def test_score_run_ranks_the_paper_latent_among_shared_latents_only():
    rng = np.random.default_rng(1)
    n = 3000
    condition = rng.random(n) < .25
    codes = np.where(rng.random((n, 200)) < .1, 1., 0.).astype(np.float32)
    codes[:, 52] = condition * (1. + rng.random(n))
    codes[:, 195] = condition * 5.  # A session-specific latent must not affect the shared rank.
    conditions = {"wheel": (condition, np.ones(n, dtype=bool))}
    _, rows = score_run(codes, conditions, 192, 192)
    assert rows[0]["latent"] == 52 and rows[0]["auroc_rank_among_shared"] == 1
    assert rows[0]["coverage_at_paper_threshold"] == 0.
    assert abs(rows[0]["other_shared_auroc_median"] - .5) < .05


def test_gpu_batches_equal_streamed_batches():
    rng = np.random.default_rng(2)
    counts = rng.poisson(1., (400, 5)).astype(np.uint16)
    starts = np.arange(0, 380, 3)
    mean, std = counts.mean(0).astype(np.float32), counts.std(0).astype(np.float32)
    reference = WindowBatches(counts, starts, mean, std, 16, 4, 0)
    fast = GpuWindowBatches(reference, "cpu")
    indices = np.array([0, 5, 17, 126])
    a, b = reference.batch(indices), fast.batch(indices)
    torch.testing.assert_close(a.values, b.values)
    assert torch.equal(a.source_indices, b.source_indices) and torch.equal(a.occurrence_mask, b.occurrence_mask)
    first = [batch.window_index.clone() for batch in fast.fresh(4, 0, 16)]
    again = [batch.window_index.clone() for batch in fast.fresh(4, 0, 16)]
    assert len(first) == 4 and all(torch.equal(x, y) for x, y in zip(first, again))


def test_identity_and_condition_transfer_detect_preserved_and_permuted_latents():
    rng = np.random.default_rng(3)
    reference = rng.normal(0, .1, (20, 12))
    reference[np.arange(12), np.arange(12)] += 1.  # Latent i separates condition i.
    same = identity_transfer(reference, reference + rng.normal(0, .02, reference.shape))
    assert same["top1"] == 1. and same["median_rank"] == 1.
    order = np.roll(np.arange(20), 1)
    moved = identity_transfer(reference, reference[order])
    assert moved["top1"] < .2
    kept = condition_transfer(reference, reference + rng.normal(0, .02, reference.shape), minimum=.5)
    lost = condition_transfer(reference, reference[order], minimum=.5)
    assert kept["n_conditions"] == 12 and kept["top1"] == 1. and lost["top1"] < .2


def test_tuning_profile_is_rate_difference_inside_minus_outside():
    codes = np.zeros((10, 2), dtype=np.float32)
    codes[:4, 0] = 1.
    codes[6:, 1] = 2.
    mask = np.arange(10) < 5
    valid = np.arange(10) != 9
    profile, sizes = tuning_profile(codes, [Condition("first_half", "test", mask, valid, "")], device="cpu")
    np.testing.assert_allclose(profile[:, 0], [4 / 5 - 0., 0. - 3 / 4])
    assert sizes.tolist() == [5]


def test_peri_event_aligns_rows_to_events_and_marks_missing_windows():
    n_bins = 2000
    endpoints = np.arange(19, n_bins)
    endpoints = endpoints[(endpoints < 900) | (endpoints > 950)]  # A neural gap.
    values = np.zeros(len(endpoints), dtype=np.float32)
    values[np.isin(endpoints, [515, 516])] = 1.
    raster = peri_event(values, endpoints, np.array([500, 940, 5]), n_bins)
    assert raster.shape == (3, len(LAGS))
    assert raster[0, LAGS == 15][0] == 1. and raster[0, LAGS == 14][0] == 0.
    assert np.isnan(raster[1, LAGS == 0][0]) and np.isnan(raster[2, LAGS == -100][0])
