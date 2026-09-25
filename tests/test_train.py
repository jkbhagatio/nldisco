import numpy as np
import pandas as pd
import pytest
import torch as t
from torch.utils.data import DataLoader

from nldisco.config import DecoderConfig, EncoderConfig, LossConfig, SedConfig, TrainConfig
from nldisco.data.window import SpikeWindowDataset
from nldisco.model import build_sed
from nldisco.train import (
    elementwise_reconstruction_loss,
    evaluate_model,
    split_rows_by_trial_proportion,
    train_model,
    weighted_reconstruction_loss,
)


@pytest.mark.parametrize(
    ("loss_type", "tau", "message"),
    [("bad", 1.0, "Unknown reconstruction loss"), ("msle", float("nan"), "tau")],
)
def test_direct_loss_helper_validates_type_and_tau(loss_type, tau, message):
    with pytest.raises(ValueError, match=message):
        elementwise_reconstruction_loss(
            t.zeros(1, 1, 1),
            t.ones(1, 1, 1),
            LossConfig(type=loss_type, tau=tau, timebin_weights=[1]),
        )


def test_weighted_reconstruction_reports_each_timebin():
    target = t.zeros(2, 3, 1)
    reconstruction = t.tensor([[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]])
    cfg = LossConfig(type="mse", timebin_weights=[1, 1, 2])
    weighted, per_timebin = weighted_reconstruction_loss(target, reconstruction, cfg)
    assert per_timebin.tolist() == [1.0, 4.0, 9.0]
    assert weighted.item() == pytest.approx((1 + 4 + 18) / 4)


def test_trial_split_uses_composite_session_and_trial_identity():
    trial_ids = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    session_ids = np.asarray([10, 10, 10, 10, 20, 20, 20, 20])
    train_rows, validation_rows = split_rows_by_trial_proportion(
        trial_ids,
        session_ids,
        train_proportion=0.5,
        shuffle=False,
    )
    assert not (train_rows & validation_rows).any()
    assert train_rows.tolist() == [True, True, True, True, False, False, False, False]


def test_trial_split_keeps_both_partitions_nonempty():
    with pytest.raises(ValueError, match="At least two"):
        split_rows_by_trial_proportion(np.asarray([0, 0]))
    train, validation = split_rows_by_trial_proportion(
        np.asarray([0, 1]), train_proportion=0.01, shuffle=False
    )
    assert train.any() and validation.any()


def test_training_and_evaluation_use_full_windows():
    counts = t.rand(14, 4)
    dataset = SpikeWindowDataset(
        counts,
        seq_len=3,
        trial_ids=[0] * 7 + [1] * 7,
    )
    loader = DataLoader(dataset, batch_size=4)
    model = build_sed(SedConfig(n_neurons=4, seq_len=3, dsed_topk_map={6: 2}))
    loss_cfg = LossConfig(type="mse", timebin_weights=[1, 1, 2])
    train_model(
        model,
        loader,
        loss_cfg,
        TrainConfig(epochs=1, batch_size=4, log_frequency=100, dead_feature_window=100),
    )
    result = evaluate_model(model, loader, loss_cfg, replica_id=7)
    assert result.reconstructions.shape == result.targets.shape
    assert result.reconstructions.shape[1:] == (3, 4)
    assert result.metrics_by_lag["lag"].tolist() == [-2, -1, 0]
    assert set(result.activation_table["replica_id"]) == {7}
    assert set(result.evaluation_index.columns) == {"source_time_idx", "replica_id"}
    assert set(result.evaluation_index["replica_id"]) == {7}


def test_training_recalibrates_thresholds_with_the_finished_encoder():
    counts = t.arange(32, dtype=t.float32).reshape(8, 4) / 32
    dataset = SpikeWindowDataset(counts, seq_len=1)
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    model = build_sed(
        SedConfig(
            n_neurons=4,
            seq_len=1,
            dsed_topk_map={6: 2},
            threshold_decay=0.99,
        )
    )
    train_model(
        model,
        loader,
        LossConfig(type="mse", timebin_weights=[1]),
        TrainConfig(epochs=1, learning_rate=0.1, use_lr_schedule=False),
    )

    final_cutoffs = []
    with t.no_grad():
        for batch in loader:
            acts = t.relu(model.encoder(batch.values))
            flat = acts.reshape(-1)
            n_keep = min(acts.shape[0] * 2, flat.numel())
            final_cutoffs.append(flat.topk(n_keep).values[-1])
    expected = t.stack(final_cutoffs).mean()
    actual = model.sparsifier.inference_threshold(6)
    assert t.allclose(actual, expected)


def test_shift_equivariant_evaluation_emits_occurrence_times():
    counts = t.rand(12, 3)
    dataset = SpikeWindowDataset(
        counts,
        seq_len=5,
        stride=3,
        occurrence_support=(2, 0),
    )
    loader = DataLoader(dataset, batch_size=4)
    model = build_sed(
        SedConfig(
            n_neurons=3,
            seq_len=5,
            dsed_topk_map={6: 2},
            encoder=EncoderConfig(
                type="TransformerWindow",
                shift_equivariant=True,
                d_model=12,
                n_heads=3,
                n_layers=1,
                d_feedforward=24,
                attention_radius=1,
            ),
            decoder=DecoderConfig(temporal_kernel_len=3, temporal_alignment="causal"),
            inference_sparsity="sample_topk",
        )
    )
    result = evaluate_model(model, loader, LossConfig(type="mse", timebin_weights=[1] * 5))
    required = {"feature_time_idx", "source_time_idx", "lag_from_anchor"}
    assert required.issubset(result.activation_table)
    assert (result.activation_table["feature_time_idx"] >= 2).all()
    assert (
        result.activation_table["lag_from_anchor"]
        == result.activation_table["feature_time_idx"] - 4
    ).all()
    assert result.evaluation_index["source_time_idx"].tolist() == list(range(2, 12))
    with t.no_grad():
        tail_reconstruction = model(dataset[-1].values.unsqueeze(0)).reconstructions[6][0]
    assert t.allclose(result.reconstructions[-1], tail_reconstruction)


def test_weighted_loss_rejects_broadcastable_weight_length():
    with pytest.raises(ValueError, match="exactly one value"):
        weighted_reconstruction_loss(
            t.zeros(2, 3, 1),
            t.ones(2, 3, 1),
            LossConfig(type="mse", timebin_weights=[1]),
        )


def test_evaluation_is_invariant_to_batch_partitioning():
    counts = t.arange(16, dtype=t.float32).reshape(4, 4)
    dataset = SpikeWindowDataset(counts, seq_len=1)
    model = build_sed(
        SedConfig(
            n_neurons=4,
            seq_len=1,
            dsed_topk_map={6: 2},
            inference_sparsity="sample_topk",
        )
    )
    loss_cfg = LossConfig(type="mse", timebin_weights=[1])
    first = evaluate_model(model, DataLoader(dataset, batch_size=1), loss_cfg)
    second = evaluate_model(model, DataLoader(dataset, batch_size=3), loss_cfg)
    assert t.allclose(first.reconstructions, second.reconstructions)
    pd.testing.assert_frame_equal(
        first.activation_table,
        second.activation_table,
        check_exact=False,
        atol=1e-5,
        rtol=1e-5,
    )


def test_shift_evaluation_rejects_duplicate_physical_occurrences():
    counts = t.rand(6, 3)
    dataset = SpikeWindowDataset(counts, seq_len=5, stride=1)
    model = build_sed(
        SedConfig(
            n_neurons=3,
            seq_len=5,
            dsed_topk_map={6: 2},
            encoder=EncoderConfig(
                type="TransformerWindow",
                shift_equivariant=True,
                d_model=12,
                n_heads=3,
                n_layers=1,
                d_feedforward=24,
                attention_radius=1,
            ),
            decoder=DecoderConfig(temporal_kernel_len=3),
            inference_sparsity="sample_topk",
        )
    )
    with pytest.raises(ValueError, match="each physical occurrence once"):
        evaluate_model(
            model,
            DataLoader(dataset, batch_size=2),
            LossConfig(type="mse", timebin_weights=[1] * 5),
        )
