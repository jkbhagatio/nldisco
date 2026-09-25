"""Plotting utilities for SED training, reconstruction, and feature inspection."""

from typing import List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import seaborn as sns
import torch as t
from beartype import beartype
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from plotly import graph_objects as go

from nldisco.evaluation import SpectralResult
from nldisco.model.decoder import LinearWindowDecoder, TemporalConvDecoder


@beartype
def plot_latent_ablation(table: pd.DataFrame, ax: Optional[Axes] = None) -> Axes:
    """Rank fixed-code ablations; negative scores mean removal improves reconstruction."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    rows = table.loc[table.scope == "window"].sort_values("delta_r2", ascending=False)
    ax.bar(np.arange(len(rows)), rows.delta_r2, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.6)
    if len(rows) <= 30:
        ax.set_xticks(np.arange(len(rows)), rows.latent_idx.astype(str))
        ax.set_xlabel("Latent (ranked)")
    else:
        ax.set_xlabel("Latent rank")
    ax.set_ylabel("Increase in error / target variance")
    ax.set_title("Latent ablation importance (non-additive)")
    return ax


@beartype
def plot_spectral_comparison(result: SpectralResult, ax: Optional[Axes] = None) -> Axes:
    """Plot the mean of per-unit PSDs, not the PSD of population-averaged activity."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for name, psd in (
        ("Target", result.target_psd),
        ("Reconstruction", result.reconstruction_psd),
    ):
        ax.plot(result.frequencies, psd.mean(axis=1), label=name)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean output-unit PSD (model target units²/Hz)")
    ax.set_title(f"Spectral reconstruction fidelity (lag {result.lag})")
    ax.legend()
    return ax


def plot_l0_stats(l0_history: Sequence[Mapping[str, float]]) -> go.Figure:
    """Create a plot of L0 standard deviation versus mean."""

    figure = go.Figure()
    for point in l0_history:
        figure.add_trace(
            go.Scatter(
                x=[point["mean"]],
                y=[point["std"]],
                mode="markers",
                marker={"size": 10, "color": f"rgba(0, 0, 255, {point['alpha']})"},
                name=f"Step {point['step']}",
                showlegend=False,
            )
        )
    figure.update_layout(
        title="L0 standard deviation versus mean",
        xaxis_title="L0 mean",
        yaxis_title="L0 standard deviation",
    )
    return figure


def box_strip_plot(
    ax: Axes,
    data: Union[pd.DataFrame, List[float]],
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    show_legend: bool = False,
) -> Axes:
    """Create a combined boxplot and stripplot."""

    sns.boxplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        width=0.4,
        showfliers=False,
        showmeans=True,
        meanprops={"markersize": "7", "markerfacecolor": "white", "markeredgecolor": "white"},
        legend=show_legend,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        size=2,
        alpha=0.4,
        dodge=True,
        jitter=True,
        legend=False,
        ax=ax,
    )
    ax.grid(True, alpha=0.5)
    return ax


def firing_rate_hist(spike_counts: pd.DataFrame) -> Axes:
    """Create a histogram of per-neuron firing rates."""

    firing_rates = spike_counts.sum() / (spike_counts.index[-1] - spike_counts.index[0])
    ax = sns.histplot(firing_rates)
    ax.set_xlabel("Firing rate (Hz)")
    ax.set_ylabel("Neuron count")
    ax.set_title(
        "Distribution of firing rates; "
        f"n_neurons={len(firing_rates)}, total_spikes={spike_counts.sum().sum()}"
    )
    ax.grid(True, alpha=0.5)
    return ax


def plot_reconstruction_by_lag(
    metrics_by_lag: pd.DataFrame,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot the configured reconstruction loss separately at every lag."""

    required = {"lag", "reconstruction_loss"}
    if not required.issubset(metrics_by_lag):
        raise ValueError(f"metrics_by_lag must contain columns {sorted(required)}")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        metrics_by_lag["lag"],
        metrics_by_lag["reconstruction_loss"],
        marker="o",
    )
    ax.set_xlabel("Lag from window anchor (time bins)")
    ax.set_ylabel("Reconstruction loss")
    ax.set_title("SED reconstruction by lag")
    ax.grid(True, alpha=0.3)
    return ax


def decoder_feature_matrix(
    decoder: Union[LinearWindowDecoder, TemporalConvDecoder],
    latent_idx: int,
) -> np.ndarray:
    """Return one decoder feature as a [timebin, output unit] NumPy array."""

    if (
        not 0
        <= latent_idx
        < decoder.weight.shape[-1 if isinstance(decoder, LinearWindowDecoder) else 0]
    ):
        raise IndexError(f"latent_idx {latent_idx} is outside the decoder dictionary")
    with t.no_grad():
        if isinstance(decoder, TemporalConvDecoder):
            feature = decoder.weight[latent_idx].transpose(0, 1)
        else:
            feature = decoder.weight[..., latent_idx]
    return feature.detach().float().cpu().numpy()


def plot_decoder_feature(
    decoder: Union[LinearWindowDecoder, TemporalConvDecoder],
    latent_idx: int,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot one global window feature or shift-equivariant temporal kernel."""

    feature = decoder_feature_matrix(decoder, latent_idx)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(feature.T, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_xlabel("Feature-relative time bin")
    ax.set_ylabel("Output unit")
    feature_kind = "Temporal kernel" if isinstance(decoder, TemporalConvDecoder) else "Window"
    ax.set_title(f"SED latent {latent_idx}: {feature_kind.lower()} feature")
    if isinstance(decoder, TemporalConvDecoder):
        if decoder.alignment == "causal":
            lags = np.arange(-decoder.kernel_len + 1, 1)
        else:
            radius = decoder.kernel_len // 2
            lags = np.arange(-radius, radius + 1)
        ax.set_xticks(np.arange(decoder.kernel_len), labels=lags)
    else:
        lags = np.arange(-feature.shape[0] + 1, 1)
        ax.set_xticks(np.arange(feature.shape[0]), labels=lags)
    ax.figure.colorbar(image, ax=ax, label="Decoder weight")
    return ax


def plot_activation_triggered_average(
    windows: Union[np.ndarray, t.Tensor],
    occurrence_time_indices: Optional[Union[np.ndarray, t.Tensor]] = None,
    kernel_len: Optional[int] = None,
    alignment: str = "causal",
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot a global or occurrence-aligned mean spike-count pattern."""

    if isinstance(windows, np.ndarray):
        values = windows
    else:
        values = windows.detach().float().cpu().numpy()
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError("windows must contain at least one [timebin, neuron] window")
    if occurrence_time_indices is None:
        if kernel_len is not None:
            raise ValueError("kernel_len requires occurrence_time_indices")
        average = values.mean(axis=0)
        x_values = np.arange(-average.shape[0] + 1, 1)
    else:
        if kernel_len is None or kernel_len < 1:
            raise ValueError("A positive kernel_len is required for occurrence alignment")
        if isinstance(occurrence_time_indices, np.ndarray):
            occurrence_values = occurrence_time_indices
        else:
            occurrence_values = occurrence_time_indices.detach().cpu().numpy()
        if (
            not np.isfinite(occurrence_values).all()
            or not np.equal(occurrence_values, np.floor(occurrence_values)).all()
        ):
            raise ValueError("Occurrence time indices must be finite integers")
        occurrence_indices = np.asarray(occurrence_values, dtype=int)
        if occurrence_indices.shape != (values.shape[0],):
            raise ValueError("Provide one occurrence time index per window")
        if alignment == "causal":
            left, right = kernel_len - 1, 0
            x_values = np.arange(-left, 1)
        elif alignment == "center":
            if kernel_len % 2 == 0:
                raise ValueError("Centered occurrence kernels must have odd length")
            left = right = kernel_len // 2
            x_values = np.arange(-left, right + 1)
        else:
            raise ValueError(f"Unknown occurrence alignment: {alignment}")
        aligned = []
        for window, occurrence_idx in zip(values, occurrence_indices):
            start = occurrence_idx - left
            stop = occurrence_idx + right + 1
            if start < 0 or stop > window.shape[0]:
                raise ValueError("An occurrence does not have complete kernel support")
            aligned.append(window[start:stop])
        average = np.stack(aligned).mean(axis=0)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(average.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Neuron")
    ax.set_title("Activation-triggered spike-count average")
    ax.set_xticks(np.arange(len(x_values)), labels=x_values)
    ax.figure.colorbar(image, ax=ax, label="Mean spike count")
    return ax
