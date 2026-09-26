import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch as t

matplotlib.use("Agg")

from nldisco.model.decoder import LinearWindowDecoder, TemporalConvDecoder
from nldisco.plot import (
    decoder_feature_matrix,
    plot_activation_triggered_average,
    plot_decoder_feature,
    plot_reconstruction_by_lag,
)


def test_decoder_features_have_timebin_by_neuron_shape():
    linear = LinearWindowDecoder(4, 5, 3, "relu")
    temporal = TemporalConvDecoder(4, 3, 3, "causal", "relu")
    assert decoder_feature_matrix(linear, 1).shape == (5, 3)
    assert decoder_feature_matrix(temporal, 1).shape == (3, 3)
    linear_ax = plot_decoder_feature(linear, 1)
    assert [tick.get_text() for tick in linear_ax.get_xticklabels()] == [
        "-4",
        "-3",
        "-2",
        "-1",
        "0",
    ]
    ax = plot_decoder_feature(temporal, 1)
    assert ax.get_title().startswith("SED latent")
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["-2", "-1", "0"]


def test_reconstruction_by_lag_plot():
    metrics = pd.DataFrame({"lag": [-2, -1, 0], "reconstruction_loss": [0.3, 0.2, 0.1]})
    ax = plot_reconstruction_by_lag(metrics)
    assert list(ax.lines[0].get_xdata()) == [-2, -1, 0]


def test_activation_triggered_average_aligns_occurrences():
    windows = t.zeros(2, 6, 1)
    windows[0, 1:4, 0] = t.tensor([1.0, 2.0, 3.0])
    windows[1, 3:6, 0] = t.tensor([1.0, 2.0, 3.0])
    ax = plot_activation_triggered_average(
        windows,
        occurrence_time_indices=t.tensor([3, 5]),
        kernel_len=3,
        alignment="causal",
    )
    plotted = ax.images[0].get_array()
    assert np.allclose(plotted, [[1.0, 2.0, 3.0]])


def test_global_activation_average_uses_anchor_relative_lags():
    ax = plot_activation_triggered_average(t.ones(2, 3, 1))
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["-2", "-1", "0"]


def test_activation_triggered_average_accepts_tensor_indices_without_truncation():
    windows = t.ones(1, 3, 1)
    plot_activation_triggered_average(
        windows,
        occurrence_time_indices=t.tensor([2]),
        kernel_len=2,
    )
    with pytest.raises(ValueError, match="finite integers"):
        plot_activation_triggered_average(
            windows,
            occurrence_time_indices=t.tensor([1.5]),
            kernel_len=2,
        )
