"""Encoders that map spike-count windows to SED feature scores."""

import math
from typing import Optional

import torch as t
from torch import Tensor, nn


class FlatWindowEncoder(nn.Module):
    """Encode a complete timebin-by-neuron window with one linear map."""

    code_layout = "global"

    def __init__(self, n_neurons: int, seq_len: int, n_features: int) -> None:
        super().__init__()
        self.n_neurons = n_neurons
        self.seq_len = seq_len
        self.projection = nn.Linear(seq_len * n_neurons, n_features)
        nn.init.kaiming_normal_(self.projection.weight, nonlinearity="relu")
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        if padding_mask is not None:
            raise ValueError("FlatWindowEncoder does not support padded windows")
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.n_neurons):
            raise ValueError(
                f"Expected [batch, {self.seq_len}, {self.n_neurons}], got {tuple(x.shape)}"
            )
        return self.projection(x.flatten(start_dim=1))


class TemporalAttentionBlock(nn.Module):
    """Pre-norm temporal self-attention with optional learned relative bias."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_feedforward: int,
        dropout: float,
        max_seq_len: int,
        relative_positions: bool,
        causal: bool,
        attention_radius: Optional[int],
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.relative_positions = relative_positions
        self.causal = causal
        self.attention_radius = attention_radius
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_feedforward = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        if relative_positions:
            self.relative_bias = nn.Parameter(t.zeros(n_heads, 2 * max_seq_len - 1))
        else:
            self.register_parameter("relative_bias", None)

    def _attention_mask(self, x: Tensor) -> Optional[Tensor]:
        batch_size, seq_len, _ = x.shape
        mask = None
        relative = None
        if self.relative_bias is not None or self.attention_radius is not None:
            positions = t.arange(seq_len, device=x.device)
            relative = positions[None, :] - positions[:, None]
        if self.relative_bias is not None:
            assert relative is not None
            relative = relative.clamp(-self.max_seq_len + 1, self.max_seq_len - 1)
            relative_indices = relative + self.max_seq_len - 1
            mask = self.relative_bias[:, relative_indices]
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
            mask = mask.reshape(batch_size * self.n_heads, seq_len, seq_len)
            mask = mask.to(dtype=x.dtype)
        disallowed = None
        if self.attention_radius is not None:
            assert relative is not None
            if self.causal:
                disallowed = (relative > 0) | (relative < -self.attention_radius)
            else:
                disallowed = relative.abs() > self.attention_radius
        elif self.causal:
            disallowed = t.triu(
                t.ones(seq_len, seq_len, dtype=t.bool, device=x.device), diagonal=1
            )
        if disallowed is not None:
            if mask is None:
                mask = t.zeros(seq_len, seq_len, dtype=x.dtype, device=x.device)
            mask = mask.masked_fill(disallowed, -t.inf)
        return mask

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        normalized = self.norm_attention(x)
        additive_padding_mask = None
        if padding_mask is not None:
            additive_padding_mask = t.zeros_like(padding_mask, dtype=normalized.dtype)
            additive_padding_mask = additive_padding_mask.masked_fill(padding_mask, -t.inf)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=additive_padding_mask,
            attn_mask=self._attention_mask(normalized),
            need_weights=False,
        )
        x = x + self.dropout(attended)
        x = x + self.dropout(self.feedforward(self.norm_feedforward(x)))
        return x


class TransformerWindowEncoder(nn.Module):
    """Encode population-state tokens with temporal self-attention."""

    def __init__(
        self,
        n_neurons: int,
        seq_len: int,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_feedforward: int,
        dropout: float,
        causal: bool,
        shift_equivariant: bool,
        attention_radius: int,
    ) -> None:
        super().__init__()
        self.n_neurons = n_neurons
        self.seq_len = seq_len
        self.shift_equivariant = shift_equivariant
        self.causal = causal
        self.n_layers = n_layers
        self.attention_radius = attention_radius if shift_equivariant else None
        self.code_layout = "temporal" if shift_equivariant else "global"
        self.input_projection = nn.Linear(n_neurons, d_model)
        if shift_equivariant:
            self.register_parameter("absolute_position", None)
        else:
            self.absolute_position = nn.Parameter(t.empty(1, seq_len, d_model))
            nn.init.normal_(self.absolute_position, std=1 / math.sqrt(d_model))
        self.blocks = nn.ModuleList(
            [
                TemporalAttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_feedforward=d_feedforward,
                    dropout=dropout,
                    max_seq_len=seq_len,
                    relative_positions=shift_equivariant,
                    causal=causal,
                    attention_radius=self.attention_radius,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.feature_projection = nn.Linear(d_model, n_features)
        nn.init.kaiming_normal_(self.feature_projection.weight, nonlinearity="relu")
        nn.init.zeros_(self.feature_projection.bias)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.n_neurons):
            raise ValueError(
                f"Expected [batch, {self.seq_len}, {self.n_neurons}], got {tuple(x.shape)}"
            )
        hidden = self.input_projection(x)
        if self.absolute_position is not None:
            hidden = hidden + self.absolute_position
        for block in self.blocks:
            hidden = block(hidden, padding_mask)
        hidden = self.output_norm(hidden)
        if not self.shift_equivariant:
            hidden = hidden[:, -1]
        return self.feature_projection(hidden)


# Preserve imports and full-module checkpoints created before the public rename.
TemporalTransformerEncoder = TransformerWindowEncoder
