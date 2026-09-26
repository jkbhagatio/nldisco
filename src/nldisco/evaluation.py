"""Optional reconstruction diagnostics, also usable with an already trained model.

Ablations hold all other sparse activations fixed (no re-encoding or re-sparsifying).
Spectra describe chronologically ordered, aligned activity in model target units.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from beartype import beartype
from einops import rearrange, repeat
from jaxtyping import Float, Integer, jaxtyped
from scipy.signal import welch
from torch import Tensor
from torch.utils.data import DataLoader

from nldisco.config import SedConfig
from nldisco.data.window import WindowSample
from nldisco.model import Sed


@dataclass
class AblationConfig:
    """Fixed-code ablation at one nested level; None selects the largest/all latents."""

    enabled: bool = False
    level: Optional[int] = None
    latents: Optional[list[int]] = None
    latent_batch_size: int = 8


@dataclass
class SpectralConfig:
    """Welch PSD settings; bin size and segment duration are in seconds."""

    enabled: bool = False
    level: Optional[int] = None
    bin_size: Optional[float] = None
    segment_duration: float = 2.0
    lag: int = 0
    bands: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Opt-in diagnostics, independent of the training objective."""

    ablation: AblationConfig = field(default_factory=AblationConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)


@dataclass
class SpectralResult:
    """PSD arrays are frequency-by-unit; undefined relative errors are NaN."""

    frequencies: np.ndarray
    target_psd: np.ndarray
    reconstruction_psd: np.ndarray
    metrics: pd.DataFrame
    band_power: pd.DataFrame
    n_segments: int
    n_samples_used: int
    n_samples_excluded: int
    bin_size: float
    segment_duration: float
    lag: int


@dataclass
class DiagnosticResult:
    """Ablation rows include lag-specific and time-weighted window scores."""

    ablation: Optional[pd.DataFrame] = None
    spectral: Optional[SpectralResult] = None


def _level(model: Sed, requested: Optional[int]) -> int:
    level = model.cfg.n_features if requested is None else requested
    if level not in model.cfg.dsed_topk_map:
        raise ValueError(f"Unknown Matryoshka level: {level}")
    return level


def _diagnostic_values(
    batch: WindowSample, model: Sed, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Validate paired shapes before encoding inputs or comparing output populations."""
    inputs = batch.values.to(device=device, dtype=model.cfg.dtype)
    targets = batch.target.to(device=device, dtype=model.cfg.dtype)
    for name, values, width in (
        ("inputs", inputs, model.cfg.n_neurons),
        ("targets", targets, model.cfg.n_output_neurons),
    ):
        if values.ndim != 3 or values.shape[1:] != (model.cfg.seq_len, width):
            raise ValueError(
                f"Evaluation {name} must have shape [batch, {model.cfg.seq_len}, {width}]"
            )
        if not torch.isfinite(values).all():
            raise ValueError(f"Evaluation {name} must be finite")
    if len(inputs) != len(targets):
        raise ValueError("Evaluation inputs and targets must contain the same number of windows")
    return inputs, targets


def _spectral_settings(config: SpectralConfig) -> tuple[float, int]:
    dt = config.bin_size
    if dt is None or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Spectral analysis requires a finite positive bin_size in seconds")
    duration = config.segment_duration
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("segment_duration must be finite and positive")
    nperseg = round(duration / dt)
    if nperseg < 4 or not np.isclose(nperseg * dt, duration):
        raise ValueError("segment_duration must be an integer number of bins, at least four")
    for name, band in config.bands.items():
        if len(band) != 2 or not np.isfinite(band).all() or not 0 <= band[0] < band[1] <= 0.5 / dt:
            raise ValueError(f"Band {name!r} must have 0 <= low < high <= Nyquist")
    return dt, nperseg


@beartype
def validate_diagnostics(config: EvaluationConfig, model: SedConfig) -> None:
    """Validate enabled options before making diagnostic inference passes."""
    if config.ablation.enabled:
        level = model.n_features if config.ablation.level is None else config.ablation.level
        if level not in model.dsed_topk_map:
            raise ValueError(f"Unknown Matryoshka level: {level}")
        ids = config.ablation.latents
        if config.ablation.latent_batch_size < 1:
            raise ValueError("latent_batch_size must be positive")
        if ids is not None and (
            not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= level for i in ids)
        ):
            raise ValueError("latents must be unique indices within the selected level")
    if config.spectral.enabled:
        if config.spectral.level is not None and config.spectral.level not in model.dsed_topk_map:
            raise ValueError(f"Unknown Matryoshka level: {config.spectral.level}")
        _spectral_settings(config.spectral)
        if not 1 - model.seq_len <= config.spectral.lag <= 0:
            raise ValueError("lag must select a bin in the reconstruction window")


@jaxtyped(typechecker=beartype)
def ablation_squared_errors(
    model: Sed,
    codes: Float[Tensor, "batch ... feature"],
    targets: Float[Tensor, "batch time unit"],
    latents: list[int],
    latent_batch_size: int = 8,
) -> Float[Tensor, "latent time"]:
    """Sum squared errors over examples and units after each fixed-code ablation.

    Only decoder calls are repeated, in bounded groups. All occurrences of a temporal
    latent are removed; bias and output nonlinearity are retained. No gradients are used.
    """
    if latent_batch_size < 1 or not latents or any(i < 0 or i >= codes.shape[-1] for i in latents):
        raise ValueError("Provide valid latent indices and a positive latent_batch_size")
    if targets.shape[1:] != (model.cfg.seq_len, model.cfg.n_output_neurons):
        raise ValueError("Ablation targets must match the decoder time and output unit dimensions")
    results = []
    with torch.no_grad():
        for start in range(0, len(latents), latent_batch_size):
            ids = latents[start : start + latent_batch_size]
            ablated = repeat(codes, "b ... -> a b ...", a=len(ids)).clone()
            for j, latent in enumerate(ids):
                ablated[j, ..., latent] = 0
            reconstruction = model.decode(rearrange(ablated, "a b ... -> (a b) ..."))
            reconstruction = rearrange(reconstruction, "(a b) s n -> a b s n", a=len(ids))
            results.append((reconstruction.double() - targets.double()).square().sum((1, 3)))
    return torch.cat(results)


@beartype
def latent_ablation(
    model: Sed,
    loader: DataLoader,
    config: AblationConfig,
    timebin_weights: list[Union[float, int]],
) -> pd.DataFrame:
    """Measure SSE increases and delta R² with stable streaming target moments.

    SST centers each unit at each lag across evaluation windows. Window delta R²
    divides the weighted sum of SSE increases by weighted SST, rather than averaging
    per-lag ratios. Constant targets have undefined delta R² (NaN), but valid MSE.
    Scores can be negative and do not sum to total explained variance.
    """
    validate_diagnostics(
        EvaluationConfig(
            ablation=AblationConfig(True, config.level, config.latents, config.latent_batch_size)
        ),
        model.cfg,
    )
    level = _level(model, config.level)
    ids = list(range(level)) if config.latents is None else config.latents
    weights = np.asarray(timebin_weights, dtype=np.float64)
    if (
        weights.shape != (model.cfg.seq_len,)
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("Provide one finite positive weight per time bin")
    weights /= weights.sum()
    device = next(model.parameters()).device
    count, mean, m2, base_sse, removed_sse = 0, None, None, None, None
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                inputs, target = _diagnostic_values(batch, model, device)
                output = model(inputs)
                reconstruction = output.reconstructions[level]
                if not torch.isfinite(reconstruction).all():
                    raise ValueError("Reconstructions must be finite")
                block = target.double()
                block_mean = block.mean(0)
                block_m2 = (block - block_mean).square().sum(0)
                base = (reconstruction.double() - block).square().sum((0, 2))
                removed = ablation_squared_errors(
                    model, output.sparse_acts[level], target, ids, config.latent_batch_size
                )
                if not torch.isfinite(removed).all():
                    raise ValueError("Ablated reconstruction errors must be finite")
                if count == 0:
                    mean, m2, base_sse, removed_sse = block_mean, block_m2, base, removed
                else:
                    delta = block_mean - mean
                    m2 += block_m2 + delta.square() * count * len(block) / (count + len(block))
                    mean += delta * len(block) / (count + len(block))
                    base_sse += base
                    removed_sse += removed
                count += len(block)
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("No evaluation windows")
    sst = m2.sum(-1).cpu().numpy()
    base = base_sse.cpu().numpy()
    removed = removed_sse.cpu().numpy()
    rows = []
    for j, latent in enumerate(ids):
        for time_idx in [*range(model.cfg.seq_len), None]:
            total = float(sst @ weights) if time_idx is None else sst[time_idx]
            full = float(base @ weights) if time_idx is None else base[time_idx]
            without = float(removed[j] @ weights) if time_idx is None else removed[j, time_idx]
            denominator = count * model.cfg.n_output_neurons
            rows.append(
                dict(
                    level=level,
                    latent_idx=latent,
                    scope="window" if time_idx is None else "lag",
                    lag=None if time_idx is None else time_idx - model.cfg.seq_len + 1,
                    n_windows=count,
                    full_mse=full / denominator,
                    ablated_mse=without / denominator,
                    delta_mse=(without - full) / denominator,
                    delta_r2=(without - full) / total if total > 0 else np.nan,
                    target_sst=total,
                )
            )
    return pd.DataFrame(rows)


@jaxtyped(typechecker=beartype)
def spectral_comparison(
    target: Float[np.ndarray, "time unit"],
    reconstruction: Float[np.ndarray, "time unit"],
    source_indices: Integer[np.ndarray, "time"],  # noqa: F821
    config: SpectralConfig,
    *,
    trial_codes: Optional[Integer[np.ndarray, "time"]] = None,  # noqa: F821
    session_codes: Optional[Integer[np.ndarray, "time"]] = None,  # noqa: F821
    timestamps: Optional[Float[np.ndarray, "time"]] = None,  # noqa: F821
) -> SpectralResult:
    """Compare per-unit PSDs without crossing gaps, trials, or sessions.

    Sorts by session/trial/source identity and rejects duplicate identities. Welch uses
    Hann windows, constant detrending, and 50% overlap. Only complete segments contribute;
    segment counts weight the PSD average across contiguous runs. Band powers sum PSD
    bins in [low, high), including Nyquist when it is the band's upper edge. Relative
    L1 error integrates |P_reconstruction-P_target| / integral(P_target).
    """
    dt, nperseg = _spectral_settings(config)
    if not target.size or not np.isfinite(target).all() or not np.isfinite(reconstruction).all():
        raise ValueError("Spectral inputs must be nonempty and finite")
    n = len(target)
    trials = np.zeros(n, dtype=np.int64) if trial_codes is None else trial_codes
    sessions = np.zeros(n, dtype=np.int64) if session_codes is None else session_codes
    if timestamps is not None and not np.isfinite(timestamps).all():
        raise ValueError("Spectral timestamps must be finite")
    order = np.lexsort((source_indices, trials, sessions))
    sources, trials, sessions = source_indices[order], trials[order], sessions[order]
    same_group = (np.diff(trials) == 0) & (np.diff(sessions) == 0)
    if ((np.diff(sources) == 0) & same_group).any():
        raise ValueError("Each physical time bin must occur once in spectral inputs")
    contiguous = (np.diff(sources) == 1) & same_group
    if timestamps is not None:
        contiguous &= np.isclose(np.diff(timestamps[order]), dt, rtol=1e-5, atol=1e-8)
    boundaries = np.r_[0, np.flatnonzero(~contiguous) + 1, n]
    psds = np.zeros((2, nperseg // 2 + 1, target.shape[1]), dtype=np.float64)
    n_segments = n_used = 0
    hop = nperseg - nperseg // 2
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        length = stop - start
        if length < nperseg:
            continue
        segments = 1 + (length - nperseg) // hop
        used = nperseg + (segments - 1) * hop
        indices = order[start : start + used]
        for i, values in enumerate((target, reconstruction)):
            frequencies, psd = welch(
                np.asarray(values[indices], dtype=np.float64),
                fs=1 / dt,
                window="hann",
                nperseg=nperseg,
                noverlap=nperseg // 2,
                detrend="constant",
                axis=0,
                scaling="density",
            )
            psds[i] += segments * psd
        n_segments += segments
        n_used += used
    if n_segments == 0:
        raise ValueError(
            "No complete spectral segments; use consecutive bins or a shorter duration"
        )
    psds /= n_segments
    df = 1 / (nperseg * dt)
    target_power, reconstruction_power = psds.sum(axis=1) * df
    discrepancy = np.abs(psds[1] - psds[0]).sum(axis=0) * df
    relative = np.divide(
        discrepancy, target_power, out=np.full_like(discrepancy, np.nan), where=target_power > 0
    )
    metrics = pd.DataFrame(
        dict(
            unit_idx=np.arange(target.shape[1]),
            target_power=target_power,
            reconstruction_power=reconstruction_power,
            absolute_psd_error=discrepancy,
            relative_psd_error=relative,
        )
    )
    bands = []
    for name, (low, high) in config.bands.items():
        selected = (frequencies >= low) & (frequencies < high)
        if np.isclose(high, 0.5 / dt):
            selected |= np.isclose(frequencies, high)
        if not selected.any():
            raise ValueError(f"Band {name!r} contains no frequency bins at this segment duration")
        powers = psds[:, selected].sum(axis=1) * df
        for unit in range(target.shape[1]):
            bands.append(
                dict(
                    band=name,
                    unit_idx=unit,
                    low_hz=low,
                    high_hz=high,
                    target_power=powers[0, unit],
                    reconstruction_power=powers[1, unit],
                    power_difference=powers[1, unit] - powers[0, unit],
                )
            )
    return SpectralResult(
        frequencies,
        psds[0],
        psds[1],
        metrics,
        pd.DataFrame(bands),
        int(n_segments),
        int(n_used),
        int(n - n_used),
        dt,
        nperseg * dt,
        config.lag,
    )


@beartype
def spectral_from_model(model: Sed, loader: DataLoader, config: SpectralConfig) -> SpectralResult:
    """Reconstruct one fixed lag per window; loader must cover consecutive source bins.

    SpikeWindowDataset timestamps are used when available. Custom datasets must encode
    temporal gaps in source indices or use spectral_comparison with explicit timestamps.
    """
    validate_diagnostics(
        EvaluationConfig(
            spectral=SpectralConfig(
                True,
                config.level,
                config.bin_size,
                config.segment_duration,
                config.lag,
                config.bands,
            )
        ),
        model.cfg,
    )
    level = _level(model, config.level)
    idx = model.cfg.seq_len - 1 + config.lag
    device = next(model.parameters()).device
    targets, reconstructions, sources, trials, sessions, times = [], [], [], [], [], []
    dataset = loader.dataset
    dataset_times = getattr(dataset, "timestamps", None)
    dataset_bin_size = getattr(dataset, "expected_bin_size", None)
    if dataset_bin_size is not None and not np.isclose(dataset_bin_size, config.bin_size):
        raise ValueError("Spectral bin_size disagrees with dataset timing")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                inputs, target = _diagnostic_values(batch, model, device)
                output = model(inputs)
                targets.append(target[:, idx].double().cpu().numpy())
                reconstructions.append(
                    output.reconstructions[level][:, idx].double().cpu().numpy()
                )
                sources.append(batch.source_indices[:, idx].cpu().numpy())
                trials.append(batch.trial_code.cpu().numpy())
                sessions.append(batch.session_code.cpu().numpy())
                if dataset_times is not None:
                    starts = np.asarray(dataset.valid_starts)[batch.window_index.cpu().numpy()]
                    times.append(dataset_times[starts + idx])
    finally:
        model.train(was_training)
    if not targets:
        raise ValueError("No evaluation windows")
    return spectral_comparison(
        np.concatenate(targets),
        np.concatenate(reconstructions),
        np.concatenate(sources),
        config,
        trial_codes=np.concatenate(trials),
        session_codes=np.concatenate(sessions),
        timestamps=np.concatenate(times) if times else None,
    )


@beartype
def evaluate_diagnostics(
    model: Sed,
    loader: DataLoader,
    config: EvaluationConfig,
    timebin_weights: list[Union[float, int]],
    *,
    spectral_loader: Optional[DataLoader] = None,
) -> DiagnosticResult:
    """Run enabled diagnostics without fitting or recalibrating the model."""
    validate_diagnostics(config, model.cfg)
    return DiagnosticResult(
        ablation=latent_ablation(model, loader, config.ablation, timebin_weights)
        if config.ablation.enabled
        else None,
        spectral=spectral_from_model(
            model, loader if spectral_loader is None else spectral_loader, config.spectral
        )
        if config.spectral.enabled
        else None,
    )


@beartype
def save_diagnostics(result: DiagnosticResult, directory: Path) -> dict[str, float]:
    """Save numeric outputs and plots to a new directory; return finite log summaries."""
    from matplotlib import pyplot as plt

    from nldisco.plot import plot_latent_ablation, plot_spectral_comparison

    directory.mkdir(parents=True, exist_ok=False)
    summary = {}
    if result.ablation is not None:
        result.ablation.to_csv(directory / "latent_ablation.csv", index=False)
        values = result.ablation.loc[result.ablation.scope == "window", "delta_r2"]
        finite = values[np.isfinite(values)]
        if len(finite):
            summary["ablation/mean_delta_r2"] = float(finite.mean())
        ax = plot_latent_ablation(result.ablation)
        ax.figure.savefig(directory / "latent_ablation.pdf", bbox_inches="tight")
        plt.close(ax.figure)
    if result.spectral is not None:
        spectral = result.spectral
        np.savez_compressed(
            directory / "spectra.npz",
            frequencies_hz=spectral.frequencies,
            target_psd=spectral.target_psd,
            reconstruction_psd=spectral.reconstruction_psd,
        )
        spectral.metrics.to_csv(directory / "spectral_metrics.csv", index=False)
        if not spectral.band_power.empty:
            spectral.band_power.to_csv(directory / "band_power.csv", index=False)
        metadata = {
            k: getattr(spectral, k)
            for k in (
                "n_segments",
                "n_samples_used",
                "n_samples_excluded",
                "bin_size",
                "segment_duration",
                "lag",
            )
        }
        metadata.update(
            window="hann",
            detrend="constant",
            overlap_fraction=0.5,
            units="model target units squared per Hz",
            population="output",
            n_output_units=spectral.target_psd.shape[1],
            axes=["frequency", "unit"],
        )
        (directory / "spectral_metadata.json").write_text(json.dumps(metadata, indent=2))
        values = spectral.metrics.relative_psd_error
        finite = values[np.isfinite(values)]
        if len(finite):
            summary["spectral/mean_relative_psd_error"] = float(finite.mean())
        summary["spectral/n_segments"] = float(spectral.n_segments)
        ax = plot_spectral_comparison(spectral)
        ax.figure.savefig(directory / "spectral_comparison.pdf", bbox_inches="tight")
        plt.close(ax.figure)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary
