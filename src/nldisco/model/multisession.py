"""Multi-session SEDs: shared latents with per-session input and output mappings.

Units recorded in different sessions do not correspond, so every session owns an
input mapping from its units into the shared temporal core and a decoder from the
latents back to its units. The core and the first ``D`` latents are shared by all
sessions; each session may add a small set of session-specific latents, placed
after the shared ones as an outermost Matryoshka level so that the shared levels
must still reconstruct the session on their own.
"""

# ruff: noqa: F722, F821 -- jaxtyping dimension strings are not Python names.

from typing import Dict, List, Optional

import torch as t
from beartype import beartype
from einops import einsum
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from nldisco.config import MultiSessionSedConfig, SessionConfig, canonical_encoder_type
from nldisco.model.decoder import LinearWindowDecoder, TemporalConvDecoder
from nldisco.model.encoder import TransformerWindowEncoder
from nldisco.model.sed import Sed, SedOutput
from nldisco.model.sparsify import BatchTopK


class SessionEncoder(nn.Module):
    """One session's input mapping and session-specific latents around a shared core."""

    def __init__(
        self,
        core: TransformerWindowEncoder,
        n_neurons: int,
        seq_len: int,
        n_session_features: int,
    ) -> None:
        super().__init__()
        self._core = [core]  # Owned by MultiSessionSed; a list keeps it unregistered here.
        self.n_neurons = n_neurons
        self.seq_len = seq_len
        self.code_layout = core.code_layout
        d_model = core.feature_projection.in_features
        self.input_projection = nn.Linear(n_neurons, d_model)
        if n_session_features:
            self.session_projection = nn.Linear(d_model, n_session_features)
            nn.init.kaiming_normal_(self.session_projection.weight, nonlinearity="relu")
            nn.init.zeros_(self.session_projection.bias)
        else:
            self.session_projection = None

    @property
    def core(self) -> TransformerWindowEncoder:
        """The temporal core and shared feature projection used by every session."""
        return self._core[0]

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.n_neurons):
            raise ValueError(
                f"Expected [batch, {self.seq_len}, {self.n_neurons}], got {tuple(x.shape)}"
            )
        core = self.core
        hidden = self.input_projection(x)
        if core.absolute_position is not None:
            hidden = hidden + core.absolute_position
        for block in core.blocks:
            hidden = block(hidden, padding_mask)
        hidden = core.output_norm(hidden)
        if not core.shift_equivariant:
            hidden = hidden[:, -1]
        scores = core.feature_projection(hidden)
        if self.session_projection is None:
            return scores
        return t.cat([scores, self.session_projection(hidden)], dim=-1)


class MultiSessionSed(nn.Module):
    """A shared temporal core and shared latents, with one :class:`Sed` view per session.

    ``model.session(name)`` is an ordinary :class:`Sed` over that session's units,
    so single-session training, calibration and evaluation code applies to it
    unchanged. Its parameters are the session's own (input mapping, decoder and
    session-specific latents); the shared core belongs to this module.
    """

    def __init__(self, cfg: MultiSessionSedConfig) -> None:
        super().__init__()
        if canonical_encoder_type(cfg.encoder.type) != "TransformerWindow":
            raise ValueError("Multi-session SEDs require the TransformerWindow encoder")
        self.cfg = cfg
        # The core's own input projection is unused: every session supplies its own.
        self.core = TransformerWindowEncoder(
            n_neurons=cfg.encoder.d_model,
            seq_len=cfg.seq_len,
            n_features=cfg.n_shared_features,
            d_model=cfg.encoder.d_model,
            n_heads=cfg.encoder.n_heads,
            n_layers=cfg.encoder.n_layers,
            d_feedforward=cfg.encoder.d_feedforward,
            dropout=cfg.encoder.dropout,
            causal=cfg.encoder.causal,
            shift_equivariant=cfg.encoder.shift_equivariant,
            attention_radius=cfg.encoder.attention_radius,
        )
        self.core.input_projection = nn.Identity()
        self.sessions = nn.ModuleDict()
        for name, session_cfg in cfg.sessions.items():
            self._build_session(name, session_cfg)
        self.to(dtype=cfg.dtype)

    def _build_session(self, name: str, session_cfg: SessionConfig) -> None:
        cfg = self.cfg.session_sed_config(name)
        encoder = SessionEncoder(
            self.core, session_cfg.n_neurons, cfg.seq_len, session_cfg.n_session_features
        )
        sparsifier = BatchTopK(
            cfg.dsed_topk_map,
            inference_sparsity=cfg.inference_sparsity,
            threshold_decay=cfg.threshold_decay,
        )
        if cfg.encoder.shift_equivariant:
            decoder: nn.Module = TemporalConvDecoder(
                n_features=cfg.n_features,
                n_neurons=cfg.n_output_neurons,
                kernel_len=cfg.decoder.temporal_kernel_len,
                alignment=cfg.decoder.temporal_alignment,
                output_activation=cfg.decoder.output_activation,
            )
        else:
            decoder = LinearWindowDecoder(
                n_features=cfg.n_features,
                seq_len=cfg.seq_len,
                n_neurons=cfg.n_output_neurons,
                output_activation=cfg.decoder.output_activation,
            )
        self.sessions[name] = Sed(cfg, encoder, sparsifier, decoder).to(dtype=cfg.dtype)

    @classmethod
    def from_sed(cls, sed: Sed, session: str) -> "MultiSessionSed":
        """Adopt a trained single-session SED as the shared core and first session."""
        if not isinstance(sed.encoder, TransformerWindowEncoder):
            raise ValueError("Only TransformerWindow SEDs can become a shared core")
        source = sed.cfg
        cfg = MultiSessionSedConfig(
            seq_len=source.seq_len,
            dsed_topk_map=dict(source.dsed_topk_map),
            sessions={
                session: SessionConfig(
                    n_neurons=source.n_neurons, n_output_neurons=source.n_output_neurons
                )
            },
            encoder=source.encoder,
            decoder=source.decoder,
            inference_sparsity=source.inference_sparsity,
            threshold_decay=source.threshold_decay,
            dtype=source.dtype,
        )
        model = cls(cfg)
        state = sed.encoder.state_dict()
        model.sessions[session].encoder.input_projection.load_state_dict(
            {key.split(".", 1)[1]: state.pop(key) for key in list(state)
             if key.startswith("input_projection.")}
        )
        model.core.load_state_dict(state)
        model.sessions[session].decoder.load_state_dict(sed.decoder.state_dict())
        model.sessions[session].sparsifier.load_state_dict(sed.sparsifier.state_dict())
        reference = next(sed.parameters())
        return model.to(device=reference.device)

    def add_session(self, name: str, session_cfg: SessionConfig) -> Sed:
        """Add a session with fresh input mapping, decoder and session-specific latents."""
        if name in self.sessions:
            raise ValueError(f"Session already exists: {name}")
        self.cfg.sessions[name] = session_cfg
        self._build_session(name, session_cfg)
        reference = self.core.feature_projection.weight
        self.sessions[name].to(device=reference.device, dtype=reference.dtype)
        return self.sessions[name]

    def session(self, name: str) -> Sed:
        """Return the :class:`Sed` view of one session."""
        if name not in self.sessions:
            raise KeyError(f"Unknown session: {name}")
        return self.sessions[name]

    def forward(
        self,
        x: Tensor,
        session: str,
        padding_mask: Optional[Tensor] = None,
        occurrence_mask: Optional[Tensor] = None,
    ) -> SedOutput:
        return self.session(session)(x, padding_mask, occurrence_mask)

    def freeze_shared(self, sessions: Optional[List[str]] = None) -> None:
        """Freeze the shared core and, optionally, every parameter of the given sessions."""
        self.core.requires_grad_(False).eval()
        for name in sessions or []:
            self.session(name).requires_grad_(False)

    def session_parameters(self, name: str) -> List[nn.Parameter]:
        """Parameters owned by one session: input mapping, decoder, session-specific latents."""
        return list(self.session(name).parameters())

    @jaxtyped(typechecker=beartype)
    @t.no_grad()
    def init_session_from(
        self,
        name: str,
        reference: str,
        input_weights: Float[Tensor, "reference_unit new_unit"],
        output_weights: Float[Tensor, "new_unit reference_unit"],
    ) -> None:
        """Initialise one session's mappings from another through a unit correspondence.

        The input mapping feeds each reference unit's input column with the new
        units that stand in for it, and each new unit inherits the shared-latent
        decoder kernels of the reference units it corresponds to (e.g. units at the
        same probe position). Shared-level thresholds are copied; session-specific
        latents keep their random initialisation.

        Args:
            name: Session to initialise.
            reference: Session whose trained mappings are transferred.
            input_weights: Row-stochastic; row ``i`` gives the new units whose
                activity approximates reference unit ``i``.
            output_weights: Row-stochastic; row ``j`` gives the reference units
                whose decoder kernels approximate new unit ``j``.
        """
        new, old = self.session(name), self.session(reference)
        if new.cfg.n_output_neurons != new.cfg.n_neurons:
            raise ValueError("Correspondence initialisation requires matching input/output units")
        if input_weights.shape != (old.cfg.n_neurons, new.cfg.n_neurons):
            raise ValueError("input_weights must have shape [reference units, new units]")
        if output_weights.shape != (new.cfg.n_neurons, old.cfg.n_neurons):
            raise ValueError("output_weights must have shape [new units, reference units]")
        input_weights = input_weights.to(new.decoder.weight)
        output_weights = output_weights.to(new.decoder.weight)
        new.encoder.input_projection.weight.copy_(
            einsum(old.encoder.input_projection.weight, input_weights, "model ref, ref new -> model new")
        )
        new.encoder.input_projection.bias.copy_(old.encoder.input_projection.bias)
        shared = self.cfg.n_shared_features
        if isinstance(new.decoder, TemporalConvDecoder):
            new.decoder.weight[:shared] = einsum(
                old.decoder.weight[:shared], output_weights, "latent ref lag, new ref -> latent new lag"
            )
            new.decoder.bias.copy_(output_weights @ old.decoder.bias)
        else:
            new.decoder.weight[..., :shared] = einsum(
                old.decoder.weight[..., :shared], output_weights,
                "time ref latent, new ref -> time new latent",
            )
            new.decoder.bias.copy_(
                einsum(old.decoder.bias, output_weights, "time ref, new ref -> time new")
            )
        new.decoder.normalize_dictionary()
        for level in old.cfg.dsed_topk_map:
            new.sparsifier.inference_threshold(level).copy_(old.sparsifier.inference_threshold(level))


def build_multisession_sed(cfg: MultiSessionSedConfig) -> MultiSessionSed:
    """Build a multi-session SED with one input and output mapping per session."""
    return MultiSessionSed(cfg)


@jaxtyped(typechecker=beartype)
def position_correspondence(
    positions: Float[Tensor, "unit dim"],
    other_positions: Float[Tensor, "other_unit dim"],
    scale: float,
) -> Float[Tensor, "unit other_unit"]:
    """Row-stochastic unit correspondence from recording-site positions.

    Each unit is matched to the other session's units with a Gaussian weight in
    their distance (``scale`` in the units of the positions; a small scale gives
    the nearest unit). No activity or behavior is used, so this is a label-free
    alignment between sessions recorded with the same probe insertion.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    distance = t.cdist(positions.double(), other_positions.double())
    logits = -0.5 * (distance / scale) ** 2
    return t.softmax(logits, dim=1).to(positions.dtype)


def session_levels(model: MultiSessionSed, name: str) -> Dict[int, int]:
    """Return the nested ``{n_latents: top_k}`` levels of one session."""
    return dict(model.session(name).cfg.dsed_topk_map)
