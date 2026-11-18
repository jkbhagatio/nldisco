"""Multi-session SEDs: shared latents, per-session mappings and session-specific latents."""

import numpy as np
import pytest
import torch as t
from torch.utils.data import DataLoader

from nldisco.config import (
    DecoderConfig,
    EncoderConfig,
    LossConfig,
    MultiSessionSedConfig,
    SedConfig,
    SessionConfig,
    TrainConfig,
)
from nldisco.data.window import SpikeWindowDataset
from nldisco.model import MultiSessionSed, build_multisession_sed, build_sed, position_correspondence
from nldisco.train import split_rows_by_session, train_model, train_multisession_model

def _same(left: t.Tensor, right: t.Tensor) -> bool:
    """Exact equality that treats uncalibrated (NaN) thresholds as equal."""
    return t.equal(t.nan_to_num(left.float(), nan=-1.0), t.nan_to_num(right.float(), nan=-1.0))


ENCODER = dict(type="TransformerWindow", d_model=16, n_heads=4, n_layers=1, d_feedforward=32)


def _encoder(shift_equivariant: bool) -> EncoderConfig:
    return EncoderConfig(shift_equivariant=shift_equivariant, attention_radius=5, **ENCODER)


def _config(shift_equivariant: bool = True) -> MultiSessionSedConfig:
    return MultiSessionSedConfig(
        seq_len=6,
        dsed_topk_map={4: 1, 8: 2},
        sessions={
            "a": SessionConfig(n_neurons=5),
            "b": SessionConfig(n_neurons=7, n_session_features=3, session_topk=1),
        },
        encoder=_encoder(shift_equivariant),
        decoder=DecoderConfig(temporal_kernel_len=6, output_activation="none"),
    )


def _single(n_neurons: int = 5) -> SedConfig:
    return SedConfig(
        n_neurons=n_neurons, seq_len=6, dsed_topk_map={4: 1, 8: 2}, encoder=_encoder(True),
        decoder=DecoderConfig(temporal_kernel_len=6, output_activation="none"),
    )


def _loader(n_neurons: int, n_rows: int = 80, seed: int = 0) -> DataLoader:
    generator = t.Generator().manual_seed(seed)
    values = t.randn(n_rows, n_neurons, generator=generator)
    dataset = SpikeWindowDataset(values, seq_len=6, occurrence_support=(5, 0))
    return DataLoader(dataset, batch_size=16, shuffle=False)


@pytest.mark.parametrize("shift_equivariant", [True, False])
def test_sessions_have_own_units_and_shared_plus_specific_latents(shift_equivariant):
    model = build_multisession_sed(_config(shift_equivariant))
    out_a = model(t.randn(4, 6, 5), "a")
    out_b = model(t.randn(4, 6, 7), "b")
    assert set(out_a.reconstructions) == {4, 8} and set(out_b.reconstructions) == {4, 8, 11}
    assert out_a.reconstructions[8].shape == (4, 6, 5)
    assert out_b.reconstructions[11].shape == (4, 6, 7)
    assert out_b.acts.shape[-1] == 11
    assert model.session("b").decoder.weight.shape[1 if shift_equivariant else 2] in (7, 11)
    with pytest.raises(ValueError):
        model(t.randn(4, 6, 5), "b")


def test_core_is_shared_once_and_not_owned_by_session_views():
    model = build_multisession_sed(_config())
    names = [name for name, _ in model.named_parameters()]
    assert len(names) == len(set(names))
    assert not any("core" in name for name, _ in model.session("b").named_parameters())
    assert model.session("a").encoder.core is model.session("b").encoder.core is model.core
    reloaded = build_multisession_sed(_config())
    reloaded.load_state_dict(model.state_dict())
    x = t.randn(3, 6, 7)
    assert t.equal(model(x, "b").pre_acts, reloaded(x, "b").pre_acts)


def test_shared_latents_ignore_session_specific_latents():
    model = build_multisession_sed(_config())
    x = t.randn(4, 6, 7)
    before = model(x, "b").pre_acts[..., :8].clone()
    with t.no_grad():
        model.session("b").encoder.session_projection.weight.add_(1.0)
    after = model(x, "b").pre_acts
    assert t.equal(before, after[..., :8])


def test_from_sed_reproduces_single_session_model_exactly():
    sed = build_sed(_single())
    train_model(sed, _loader(5), LossConfig(timebin_weights=[1.0] * 6, type="mse"),
                TrainConfig(batch_size=16, epochs=1, log_frequency=1))
    sed.eval()
    model = MultiSessionSed.from_sed(sed, "a").eval()
    x = t.randn(8, 6, 5)
    mask = t.zeros(8, 6, dtype=t.bool)
    mask[:, -1] = True
    expected, observed = sed(x, occurrence_mask=mask), model(x, "a", occurrence_mask=mask)
    assert t.equal(expected.pre_acts, observed.pre_acts)
    for level in (4, 8):
        assert t.equal(expected.sparse_acts[level], observed.sparse_acts[level])
        assert t.equal(expected.reconstructions[level], observed.reconstructions[level])


def test_adapting_a_new_session_leaves_core_and_reference_session_unchanged():
    model = MultiSessionSed.from_sed(build_sed(_single()), "a")
    model.add_session("b", SessionConfig(n_neurons=7, n_session_features=3, session_topk=1))
    model.freeze_shared(["a"])
    frozen = {name: value.clone() for name, value in model.state_dict().items()
              if not name.startswith("sessions.b.")}
    before = {name: value.clone() for name, value in model.session("b").named_parameters()}
    train_multisession_model(
        model, {"b": _loader(7)}, LossConfig(timebin_weights=[1.0] * 6, type="mse"),
        TrainConfig(batch_size=16, epochs=2, log_frequency=1),
    )
    state = model.state_dict()
    assert all(_same(value, state[name]) for name, value in frozen.items())
    changed = [not t.equal(value, dict(model.session("b").named_parameters())[name])
               for name, value in before.items()]
    assert all(changed)
    assert all(t.isfinite(model.session("b").sparsifier.inference_threshold(level))
               for level in (4, 8, 11))


def test_single_session_trainer_on_a_session_view_keeps_the_core_frozen():
    model = build_multisession_sed(_config())
    core = {name: value.clone() for name, value in model.core.state_dict().items()}
    train_model(model.session("b"), _loader(7), LossConfig(timebin_weights=[1.0] * 6, type="mse"),
                TrainConfig(batch_size=16, epochs=1, log_frequency=1))
    assert all(t.equal(value, model.core.state_dict()[name]) for name, value in core.items())


def test_joint_training_updates_core_and_every_session():
    model = build_multisession_sed(_config())
    before = {name: value.clone() for name, value in model.named_parameters()}
    histories = train_multisession_model(
        model, {"a": _loader(5), "b": _loader(7, n_rows=50)},
        LossConfig(timebin_weights=[1.0] * 6, type="mse"),
        TrainConfig(batch_size=16, epochs=1, log_frequency=1),
    )
    after = dict(model.named_parameters())
    assert all(not t.equal(value, after[name]) for name, value in before.items()
               if "relative_bias" not in name)
    assert set(histories) == {"a", "b"}
    assert all(np.isfinite(list(history.loss.values())).all() for history in histories.values())


def test_correspondence_initialisation_copies_matched_units():
    model = MultiSessionSed.from_sed(build_sed(_single()), "a")
    model.add_session("b", SessionConfig(n_neurons=7, n_session_features=3, session_topk=1))
    reference = t.tensor([[0.0, float(depth)] for depth in range(0, 50, 10)])
    order = [4, 0, 2, 2, 1, 3, 0]
    positions = reference[order] + t.linspace(0.01, 0.07, 7)[:, None]
    output_weights = position_correspondence(positions, reference, scale=0.5)
    input_weights = position_correspondence(reference, positions, scale=0.5)
    assert t.allclose(output_weights.sum(dim=1), t.ones(7))
    assert output_weights.argmax(dim=1).tolist() == order
    assert input_weights.argmax(dim=1).tolist() == [1, 4, 2, 5, 0]
    a, b = model.session("a"), model.session("b")
    a.sparsifier.inference_threshold(8).fill_(0.5)
    model.init_session_from("b", "a", input_weights, output_weights)
    expected = a.decoder.weight[:, order]
    expected = expected / expected.norm(dim=(1, 2), keepdim=True)
    assert t.allclose(b.decoder.weight[:8], expected, atol=1e-5)
    assert t.equal(b.sparsifier.inference_threshold(8), a.sparsifier.inference_threshold(8))
    # Each reference unit's input column is driven by its matched new unit alone,
    # so matched activity gives the reference session's shared latent scores.
    x = t.randn(4, 6, 5)
    assert t.allclose(model(x, "a").pre_acts, model(x[..., order], "b").pre_acts[..., :8], atol=1e-4)
    with pytest.raises(ValueError):
        model.init_session_from("b", "a", output_weights, input_weights)


def test_windows_and_splits_respect_session_boundaries():
    values = t.randn(40, 4)
    session_ids = np.repeat(np.array(["a", "b"], dtype=object), 20)
    dataset = SpikeWindowDataset(values, seq_len=6, session_ids=session_ids)
    sources = t.stack([dataset[i].source_indices for i in range(len(dataset))])
    assert len(dataset) == 2 * (20 - 6 + 1)
    assert not ((sources.min(dim=1).values < 20) & (sources.max(dim=1).values >= 20)).any()
    train, held_out = split_rows_by_session(session_ids, ["a"])
    assert train[:20].all() and not train[20:].any() and held_out[20:].all()


def test_session_configuration_is_validated():
    with pytest.raises(ValueError):
        SessionConfig(n_neurons=5, n_session_features=3)
    with pytest.raises(ValueError):
        MultiSessionSedConfig(seq_len=6, dsed_topk_map={8: 2}, sessions={"a": SessionConfig(5)},
                              encoder=EncoderConfig(type="FlatWindow"))
    model = build_multisession_sed(_config())
    with pytest.raises(ValueError):
        model.add_session("a", SessionConfig(n_neurons=5))
    with pytest.raises(KeyError):
        model.session("missing")
