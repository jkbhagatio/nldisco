"""Composition and construction of sparse encoder-decoder models."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch as t
from torch import Tensor, nn
from torch.nn import functional as F

from nldisco.config import SedConfig, canonical_encoder_type
from nldisco.model.decoder import LinearWindowDecoder, TemporalConvDecoder
from nldisco.model.encoder import FlatWindowEncoder, TransformerWindowEncoder
from nldisco.model.sparsify import BatchTopK


@dataclass
class SedOutput:
    """Outputs for every nested SED level."""

    reconstructions: Dict[int, Tensor]
    sparse_acts: Dict[int, Tensor]
    acts: Tensor
    pre_acts: Tensor
    code_layout: str


class Sed(nn.Module):
    """A configurable encoder, sparse bottleneck, and decoder."""

    def __init__(
        self, cfg: SedConfig, encoder: nn.Module, sparsifier: nn.Module, decoder: nn.Module
    ):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.sparsifier = sparsifier
        self.decoder = decoder
        if encoder.code_layout != decoder.code_layout:
            raise ValueError(
                f"Incompatible encoder/decoder layouts: {encoder.code_layout}, {decoder.code_layout}"
            )
        self.code_layout = encoder.code_layout

    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
        occurrence_mask: Optional[Tensor] = None,
    ) -> SedOutput:
        pre_acts = self.encoder(x, padding_mask)
        acts = F.relu(pre_acts)
        acts = self._mask_invalid_temporal_occurrences(acts, padding_mask, occurrence_mask)
        sparse_acts = self.sparsifier(acts)
        reconstructions = {
            d_sed: self.decoder(level_acts) for d_sed, level_acts in sparse_acts.items()
        }
        return SedOutput(
            reconstructions=reconstructions,
            sparse_acts=sparse_acts,
            acts=acts,
            pre_acts=pre_acts,
            code_layout=self.code_layout,
        )

    def _mask_invalid_temporal_occurrences(
        self,
        acts: Tensor,
        padding_mask: Optional[Tensor],
        occurrence_mask: Optional[Tensor],
    ) -> Tensor:
        """Require complete encoder and decoder support for every occurrence."""

        if self.code_layout != "temporal":
            return acts
        seq_len = acts.shape[1]
        required_left, required_right = self.temporal_occurrence_support()

        if padding_mask is None:
            valid_timebins = t.ones(acts.shape[0], seq_len, dtype=t.bool, device=acts.device)
        else:
            if padding_mask.shape != acts.shape[:2]:
                raise ValueError("padding_mask must have shape [batch, timebin]")
            valid_timebins = ~padding_mask.to(device=acts.device, dtype=t.bool)
        valid_occurrences = t.zeros_like(valid_timebins)
        for time_idx in range(required_left, seq_len - required_right):
            support = valid_timebins[:, time_idx - required_left : time_idx + required_right + 1]
            valid_occurrences[:, time_idx] = support.all(dim=1)
        if occurrence_mask is not None:
            if occurrence_mask.shape != acts.shape[:2]:
                raise ValueError("occurrence_mask must have shape [batch, timebin]")
            valid_occurrences &= occurrence_mask.to(device=acts.device, dtype=t.bool)
        return acts * valid_occurrences.unsqueeze(-1)

    def temporal_occurrence_support(self) -> Tuple[int, int]:
        """Return required left/right context for one valid temporal occurrence."""

        if self.code_layout != "temporal":
            raise ValueError("Global SEDs do not have temporal occurrence support")
        return self.cfg.temporal_occurrence_support()

    def decode(
        self,
        sparse_acts: Tensor,
        *,
        apply_output_activation: bool = True,
        include_bias: bool = True,
    ) -> Tensor:
        return self.decoder(
            sparse_acts,
            apply_output_activation=apply_output_activation,
            include_bias=include_bias,
        )

    def constrain_dictionary(self) -> None:
        self.decoder.project_dictionary_grad()
        self.decoder.normalize_dictionary()

    def normalize_dictionary(self) -> None:
        self.decoder.normalize_dictionary()


def build_sed(cfg: SedConfig) -> Sed:
    """Build a valid SED component combination from configuration."""

    if canonical_encoder_type(cfg.encoder.type) == "FlatWindow":
        encoder = FlatWindowEncoder(
            n_neurons=cfg.n_neurons,
            seq_len=cfg.seq_len,
            n_features=cfg.n_features,
        )
    else:
        encoder = TransformerWindowEncoder(
            n_neurons=cfg.n_neurons,
            seq_len=cfg.seq_len,
            n_features=cfg.n_features,
            d_model=cfg.encoder.d_model,
            n_heads=cfg.encoder.n_heads,
            n_layers=cfg.encoder.n_layers,
            d_feedforward=cfg.encoder.d_feedforward,
            dropout=cfg.encoder.dropout,
            causal=cfg.encoder.causal,
            shift_equivariant=cfg.encoder.shift_equivariant,
            attention_radius=cfg.encoder.attention_radius,
        )

    sparsifier = BatchTopK(
        cfg.dsed_topk_map,
        inference_sparsity=cfg.inference_sparsity,
        threshold_decay=cfg.threshold_decay,
    )
    if cfg.encoder.shift_equivariant:
        decoder = TemporalConvDecoder(
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
    return Sed(cfg, encoder, sparsifier, decoder).to(dtype=cfg.dtype)
