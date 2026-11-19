"""Post hoc check of the population burst after pellet triggers (not part of the protocol).

    python -m experiments.rebuttal.change03_discovery.pellet_check --output OUT

Part 1: peri-trigger time course of spike counts at 20 ms resolution for sessions A
and B, per-unit participation, onset latencies, and relation to probe position and
unit quality. Part 2: the paper's session A wheel-activity latent (D=192, k=12,
latent 52) around pellet triggers and its scores without post-trigger windows.
"""
# ruff: noqa: F821, F722  (jaxtyping shape annotations)
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from scipy.stats import spearmanr

from experiments.rebuttal.change03_discovery import data, scoring

REBUTTAL = Path("/ceph/aeon/aeon/nldisco/rebuttal_20261118")
SWEEP_A = Path("/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20")
SESSIONS = {
    "A": (SWEEP_A / "data", REBUTTAL / "change02_multisession/unit_positions/session_a.csv"),
    "B": (REBUTTAL / "change02_multisession/session_b",
          REBUTTAL / "change02_multisession/session_b/cluster_positions.csv"),
}
LAGS = np.arange(-100, 150)  # 20 ms bins: -2 s to +3 s
BASELINE = (LAGS >= -100) & (LAGS < -50)  # -2 s to -1 s
BURST = (LAGS >= 0) & (LAGS < 25)  # 0 to 0.5 s
WHEEL_CUTOFF = 0.75 / (2 * np.pi * 4.)  # turns/s; the paper's 0.75 cm/s on a 4 cm wheel
PAPER_THRESHOLD = 3.31


@jaxtyped(typechecker=beartype)
def peri_event(counts: Float[np.ndarray, "time units"], events: Int[np.ndarray, "events"],
               valid: Bool[np.ndarray, "time"]) -> Float[np.ndarray, "events lags units"]:
    """Counts around each event with complete, valid support for every lag."""
    index = events[:, None] + LAGS[None, :]
    inside = (index.min(1) >= 0) & (index.max(1) < len(counts))
    index = index[inside]
    return counts[index[valid[index].all(1)]]


@beartype
def first_run(flags: np.ndarray, length: int = 2) -> int:
    """Index of the first run of `length` consecutive true values, or -1."""
    run = np.convolve(flags.astype(int), np.ones(length, dtype=int), "valid") == length
    return int(np.argmax(run)) if run.any() else -1


@beartype
def burst_summary(name: str, root: Path, positions: Path, output: Path) -> dict:
    """Peri-trigger population time course, unit participation and latencies."""
    counts = np.asarray(np.load(root / "counts.npy", mmap_mode="r"), dtype=np.float32)
    valid = np.asarray(np.load(root / "neural_valid.npy"), dtype=bool)
    pellets = pd.read_csv(root / "pellet_times.csv")
    events = np.floor(pellets["relative_seconds"].to_numpy() / data.BIN_SECONDS).astype(np.int64)
    trials = peri_event(counts, events, valid)
    population = trials.sum(2)  # [events, lags]
    mean_population = population.mean(0)
    pd.DataFrame(dict(seconds=LAGS * data.BIN_SECONDS, population_spikes_per_bin=mean_population,
                      sem=population.std(0) / np.sqrt(len(population)))).to_csv(
        output / f"session_{name}_population_time_course.csv", index=False)
    unit = trials.mean(0)  # [lags, units]
    base, base_sd = unit[BASELINE].mean(0), unit[BASELINE].std(0)
    ratio = unit[BURST].mean(0) / np.maximum(base, 1e-9)
    added = unit[(LAGS >= 0) & (LAGS < 13)].mean(0) - base  # extra spikes per bin, 0-0.26 s
    search = LAGS >= -5
    z = (unit[search] - base) / np.maximum(base_sd, 1e-9)
    onset = np.array([first_run(z[:, u] > 5) for u in range(unit.shape[1])])
    responsive = onset >= 0
    onset_ms = (LAGS[search][onset[responsive]] * data.BIN_SECONDS * 1000).astype(int)
    peak_ms = (LAGS[search][np.argmax(unit[search], 0)] * data.BIN_SECONDS * 1000).astype(int)
    onset_values, onset_counts = np.unique(onset_ms, return_counts=True)
    # Per-trigger: size of the burst and its onset relative to the trigger.
    trial_base = population[:, BASELINE]
    trial_ratio = population[:, (LAGS >= 0) & (LAGS < 13)].mean(1) / trial_base.mean(1)
    trial_z = (population[:, search] - trial_base.mean(1, keepdims=True)) \
        / np.maximum(trial_base.std(1, keepdims=True), 1e-9)
    trial_onset = np.array([first_run(row > 5) for row in trial_z])
    trial_onset_ms = LAGS[search][trial_onset[trial_onset >= 0]] * data.BIN_SECONDS * 1000
    table = pd.read_csv(positions)
    label = table["ks_label"].to_numpy()
    depth = table["y_um"].to_numpy(dtype=float)
    if len(table) != counts.shape[1]:
        raise ValueError("Cluster position table does not match the count columns")
    terciles = np.quantile(depth, [1 / 3, 2 / 3])
    band = np.digitize(depth, terciles)
    window = (LAGS >= -10) & (LAGS < 60)
    result = dict(
        session=name, triggers_used=int(len(trials)), triggers_total=int(len(events)),
        units=int(counts.shape[1]),
        population_baseline=float(mean_population[BASELINE].mean()),
        population_peak=float(mean_population.max()),
        population_peak_ms=int(LAGS[np.argmax(mean_population)] * 20),
        population_by_20ms={f"{int(lag * 20):+d}": round(float(v), 1)
                            for lag, v in zip(LAGS[window], mean_population[window])},
        population_coarse={f"{a * .02:+.1f}..{b * .02:+.1f}s": round(float(
            mean_population[(LAGS >= a) & (LAGS < b)].mean()), 1)
            for a, b in ((-100, -50), (-50, -10), (60, 75), (75, 100), (100, 150))},
        units_ratio_ge_1p5=int((ratio >= 1.5).sum()), units_ratio_ge_2=int((ratio >= 2).sum()),
        units_ratio_ge_5=int((ratio >= 5).sum()), units_ratio_le_0p67=int((ratio <= 1 / 1.5).sum()),
        ratio_quartiles=[round(float(q), 2) for q in np.quantile(ratio, [.25, .5, .75])],
        units_with_onset=int(responsive.sum()),
        onset_ms_histogram={int(v): int(c) for v, c in zip(onset_values, onset_counts)},
        onset_ms_median=float(np.median(onset_ms)) if len(onset_ms) else None,
        onset_ms_iqr=[float(q) for q in np.quantile(onset_ms, [.25, .75])] if len(onset_ms) else None,
        peak_ms_quartiles=[float(q) for q in np.quantile(peak_ms[responsive], [.25, .5, .75])],
        triggers_with_3x_burst=int((trial_ratio >= 3).sum()),
        trigger_ratio_quartiles=[round(float(q), 2) for q in np.quantile(trial_ratio, [.25, .5, .75])],
        trigger_onset_ms_histogram={int(v): int(c) for v, c in zip(
            *np.unique(trial_onset_ms, return_counts=True))},
        triggers_with_onset=int((trial_onset >= 0).sum()),
        by_quality={kind: dict(units=int((label == kind).sum()),
                               ratio_median=round(float(np.median(ratio[label == kind])), 2),
                               ratio_ge_2=int((ratio[label == kind] >= 2).sum()),
                               with_onset=int(responsive[label == kind].sum()),
                               added_spikes_median=round(float(np.median(added[label == kind])), 3))
                    for kind in np.unique(label)},
        by_depth_tercile=[dict(depth_um=[round(float(depth[band == b].min())),
                                         round(float(depth[band == b].max()))],
                               units=int((band == b).sum()),
                               ratio_median=round(float(np.median(ratio[band == b])), 2),
                               ratio_ge_2=int((ratio[band == b] >= 2).sum()),
                               with_onset=int(responsive[band == b].sum()),
                               onset_ms_median=float(np.median(
                                   LAGS[search][onset[(band == b) & responsive]] * 20))
                               if ((band == b) & responsive).any() else None)
                          for b in range(3)],
        spearman_ratio_vs_depth=[round(float(v), 3) for v in spearmanr(ratio, depth)],
        spearman_added_vs_baseline_rate=[round(float(v), 3) for v in spearmanr(added, base)],
        spearman_ratio_vs_baseline_rate=[round(float(v), 3) for v in spearmanr(ratio, base)],
        added_spikes_per_bin_quartiles=[round(float(q), 3) for q in np.quantile(added, [.25, .5, .75])],
        baseline_spikes_per_bin_quartiles=[round(float(q), 3) for q in np.quantile(base, [.25, .5, .75])])
    pd.DataFrame(dict(cluster_id=table["cluster_id"], ks_label=label, depth_um=depth,
                      baseline_spikes_per_bin=base, ratio_0_to_500ms=ratio,
                      added_spikes_per_bin_0_to_260ms=added,
                      onset_ms=np.where(responsive, LAGS[search][np.maximum(onset, 0)] * 20, np.nan),
                      peak_ms=peak_ms)).to_csv(output / f"session_{name}_units.csv", index=False)
    return result


@beartype
def wheel_latent(output: Path) -> dict:
    """The paper's wheel-activity latent around pellet triggers (session A)."""
    root = SWEEP_A / "data"
    behavior = pd.read_parquet(root / "behavior.parquet", columns=["wheel_speed", "wheel_valid"])
    endpoints = np.load(SWEEP_A / "d192/k12_seed0/endpoint_bins.npy")
    z = np.asarray(np.load(SWEEP_A / "d192/k12_seed0/activations.npy", mmap_mode="r")[:, 52],
                   dtype=np.float64)
    wheel = behavior["wheel_speed"].to_numpy(dtype=float)
    support = data.complete_support(behavior["wheel_valid"].to_numpy(dtype=bool)
                                    & np.isfinite(wheel))[endpoints]
    condition = wheel[endpoints] > WHEEL_CUTOFF
    pellets = pd.read_csv(root / "pellet_times.csv")
    events = np.floor(pellets["relative_seconds"].to_numpy() / data.BIN_SECONDS).astype(np.int64)
    since = data.event_clocks(events, len(behavior))[0][endpoints]
    row_of_bin = np.full(len(behavior), -1, dtype=np.int64)
    row_of_bin[endpoints] = np.arange(len(endpoints))
    index = events[:, None] + LAGS[None, :]
    index = index[(index.min(1) >= 0) & (index.max(1) < len(behavior))]
    rows = row_of_bin[index]
    rows = rows[(rows >= 0).all(1)]
    course = pd.DataFrame(dict(
        seconds=LAGS * data.BIN_SECONDS, mean_activation=z[rows].mean(0),
        p_active_paper_threshold=(z[rows] >= PAPER_THRESHOLD).mean(0),
        p_positive=(z[rows] > 0).mean(0), p_wheel_active=condition[rows].mean(0),
        mean_wheel_speed_turns_s=wheel[endpoints][rows].mean(0)))
    course.to_csv(output / "session_A_wheel_latent52_time_course.csv", index=False)

    def block(a: float, b: float) -> dict:
        part = course[(course.seconds >= a - 1e-9) & (course.seconds < b - 1e-9)]
        return {k: round(float(part[k].mean()), 3) for k in course.columns if k != "seconds"}

    def score(keep: np.ndarray) -> dict:
        valid = support & keep
        result = scoring.scores(z[valid], PAPER_THRESHOLD, condition[valid])
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()}

    after_1 = np.isfinite(since) & (since < 1.)
    after_1p4 = np.isfinite(since) & (since < 1.4)  # any bin of the window within 1 s
    top = {}
    for name, quantile in (("top_1_percent", .99), ("top_0p1_percent", .999)):
        threshold = float(np.quantile(z, quantile))
        selected = z >= threshold
        top[name] = dict(threshold=round(threshold, 3), windows=int(selected.sum()),
                         share_within_1s_after_trigger=round(float(after_1[selected].mean()), 4),
                         share_wheel_active=round(float(condition[selected].mean()), 4))
    false_positive = support & ~condition & (z >= PAPER_THRESHOLD)
    return dict(
        triggers_used=int(len(rows)), paper_threshold=PAPER_THRESHOLD,
        wheel_cutoff_turns_s=round(float(WHEEL_CUTOFF), 5),
        share_of_all_windows_within_1s_after_trigger=round(float(after_1.mean()), 5),
        time_course_blocks={f"{a:+.1f}..{b:+.1f}s": block(a, b) for a, b in (
            (-2, -1), (-1, -.5), (-.5, 0), (0, .26), (.26, .5), (.5, 1), (1, 1.4), (1.4, 2), (2, 3))},
        top_activations=top,
        scores_all_windows=score(np.ones(len(z), dtype=bool)),
        scores_excluding_1s_after_trigger=score(~after_1),
        scores_excluding_windows_overlapping_1s_after_trigger=score(~after_1p4),
        scores_only_within_1s_after_trigger=score(after_1),
        false_positives_total=int(false_positive.sum()),
        false_positives_within_1s_after_trigger=int((false_positive & after_1).sum()),
        wheel_active_windows_total=int((support & condition).sum()),
        wheel_active_windows_within_1s_after_trigger=int((support & condition & after_1).sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(bursts=[burst_summary(name, root, positions, args.output)
                          for name, (root, positions) in SESSIONS.items()],
                  wheel_latent_session_a=wheel_latent(args.output))
    (args.output / "pellet_check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
