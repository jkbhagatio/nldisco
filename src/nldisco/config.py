"""Typed configuration for SED models and training."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import torch as t

EncoderType = Literal["FlatWindow", "TransformerWindow"]


def canonical_encoder_type(value: str) -> EncoderType:
    """Resolve current encoder names and legacy serialized configuration values."""
    names = {
        "FlatWindow": "FlatWindow", "TransformerWindow": "TransformerWindow",
        "flat": "FlatWindow", "flat_window": "FlatWindow",
        "temporal_transformer": "TransformerWindow",
    }
    if value not in names:
        raise ValueError(f"Unknown encoder type: {value}")
    return names[value]


@dataclass
class WindowConfig:
    """Configuration for constructing fixed-length spike-count windows."""

    seq_len: int
    stride: int = 1
    expected_bin_size: Optional[float] = None

    def __post_init__(self) -> None:
        if self.seq_len < 1:
            raise ValueError("seq_len must be at least 1")
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        if self.expected_bin_size is not None and (
            not math.isfinite(self.expected_bin_size) or self.expected_bin_size <= 0
        ):
            raise ValueError("expected_bin_size must be finite and positive")


@dataclass
class EncoderConfig:
    """Configuration for FlatWindow and TransformerWindow encoders."""

    type: Literal["FlatWindow", "TransformerWindow", "flat", "flat_window", "temporal_transformer"] = "FlatWindow"
    shift_equivariant: bool = False
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_feedforward: int = 512
    dropout: float = 0.0
    causal: bool = True
    attention_radius: int = 2

    def __post_init__(self) -> None:
        self.type = canonical_encoder_type(self.type)
        if self.shift_equivariant and self.type != "TransformerWindow":
            raise ValueError("shift_equivariant is only supported by TransformerWindow")
        if self.d_model < 1 or self.n_heads < 1 or self.n_layers < 1:
            raise ValueError("Transformer dimensions and layer counts must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.attention_radius < 0:
            raise ValueError("attention_radius cannot be negative")


@dataclass
class DecoderConfig:
    """Configuration for the decoder selected by the SED factory."""

    output_activation: Literal["relu", "softplus", "none"] = "relu"
    temporal_kernel_len: int = 5
    temporal_alignment: Literal["center", "causal"] = "causal"

    def __post_init__(self) -> None:
        if self.output_activation not in {"relu", "softplus", "none"}:
            raise ValueError(f"Unknown decoder output activation: {self.output_activation}")
        if self.temporal_kernel_len < 1:
            raise ValueError("temporal_kernel_len must be at least 1")
        if self.temporal_alignment not in {"center", "causal"}:
            raise ValueError(f"Unknown temporal alignment: {self.temporal_alignment}")
        if self.temporal_alignment == "center" and self.temporal_kernel_len % 2 == 0:
            raise ValueError("Centered temporal kernels must have odd length")


@dataclass
class SedConfig:
    """Architecture for one SED; output width defaults to the input unit count.

    ``n_neurons`` always describes encoder inputs. ``n_output_neurons`` controls
    decoder targets on the same time grid and with the same ``seq_len``.
    """

    n_neurons: int
    seq_len: int
    dsed_topk_map: Dict[int, int]
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    inference_sparsity: Literal["training_threshold", "sample_topk"] = "training_threshold"
    threshold_decay: float = 0.99
    dtype: t.dtype = t.float32
    n_output_neurons: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_neurons < 1 or self.seq_len < 1:
            raise ValueError("n_neurons and seq_len must be positive")
        if self.n_output_neurons is None:
            self.n_output_neurons = self.n_neurons
        if self.n_output_neurons < 1:
            raise ValueError("n_output_neurons must be positive")
        if not self.dsed_topk_map:
            raise ValueError("dsed_topk_map must contain at least one SED level")
        if any(d_sed < 1 or top_k < 1 for d_sed, top_k in self.dsed_topk_map.items()):
            raise ValueError("SED dimensions and TopK values must be positive")
        if self.inference_sparsity not in {"training_threshold", "sample_topk"}:
            raise ValueError(f"Unknown inference sparsity policy: {self.inference_sparsity}")
        if not math.isfinite(self.threshold_decay) or not 0 <= self.threshold_decay < 1:
            raise ValueError("threshold_decay must be finite and in [0, 1)")
        if not self.encoder.shift_equivariant:
            invalid = [
                (d_sed, top_k) for d_sed, top_k in self.dsed_topk_map.items() if top_k > d_sed
            ]
            if invalid:
                raise ValueError(
                    f"Global TopK values cannot exceed their SED dimensions: {invalid}"
                )
        else:
            if self.decoder.temporal_kernel_len > self.seq_len:
                raise ValueError("Shift-equivariant temporal_kernel_len cannot exceed seq_len")
            (encoder_left, encoder_right), (decoder_left, decoder_right) = (
                self._temporal_component_supports()
            )
            required_left, required_right = self.temporal_occurrence_support()
            if required_left + required_right >= self.seq_len:
                raise ValueError(
                    "Shift-equivariant attention and decoder supports leave no valid "
                    "occurrence positions in the configured sequence"
                )
            if decoder_left < encoder_left or decoder_right < encoder_right:
                raise ValueError(
                    "Shift-equivariant decoder support must cover encoder context on both sides "
                    "so every reconstructed time bin is feature-reachable"
                )
            usable_positions = self.seq_len - required_left - required_right
            invalid = [
                (d_sed, top_k)
                for d_sed, top_k in self.dsed_topk_map.items()
                if top_k > usable_positions * d_sed
            ]
            if invalid:
                raise ValueError(
                    "Temporal TopK values cannot exceed usable_positions * d_sed "
                    f"({usable_positions} usable positions): {invalid}"
                )

    def _temporal_component_supports(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return encoder and decoder context extents for temporal occurrences."""

        attention_extent = self.encoder.n_layers * self.encoder.attention_radius
        if self.encoder.causal:
            encoder_support = (attention_extent, 0)
        else:
            encoder_support = (attention_extent, attention_extent)
        if self.decoder.temporal_alignment == "causal":
            decoder_support = (self.decoder.temporal_kernel_len - 1, 0)
        else:
            radius = self.decoder.temporal_kernel_len // 2
            decoder_support = (radius, radius)
        return encoder_support, decoder_support

    def temporal_occurrence_support(self) -> Tuple[int, int]:
        """Return required left/right context for a shift-equivariant occurrence."""

        if not self.encoder.shift_equivariant:
            raise ValueError("Only shift-equivariant SEDs have temporal occurrence support")
        encoder_support, decoder_support = self._temporal_component_supports()
        return (
            max(encoder_support[0], decoder_support[0]),
            max(encoder_support[1], decoder_support[1]),
        )

    @property
    def usable_occurrence_positions(self) -> int:
        """Number of fully supported temporal positions in each input window."""

        required_left, required_right = self.temporal_occurrence_support()
        return self.seq_len - required_left - required_right

    @property
    def n_features(self) -> int:
        """Largest nested SED feature dimension."""

        return max(self.dsed_topk_map)


@dataclass
class LossConfig:
    """Full-window reconstruction objective configuration."""

    timebin_weights: List[float]
    type: Literal["mse", "msle"] = "msle"
    tau: float = 1.0
    dsed_loss_weight_map: Optional[Dict[int, float]] = None
    dead_feature_loss_weight: float = 1.0

    def validate_for(self, sed_cfg: SedConfig) -> None:
        """Validate loss settings against a concrete model shape."""

        if self.type not in {"mse", "msle"}:
            raise ValueError(f"Unknown reconstruction loss: {self.type}")
        if len(self.timebin_weights) != sed_cfg.seq_len:
            raise ValueError(
                "timebin_weights must contain exactly one positive value per configured time bin "
                f"({len(self.timebin_weights)} != {sed_cfg.seq_len})"
            )
        if any(not math.isfinite(weight) or weight <= 0 for weight in self.timebin_weights):
            raise ValueError("Every timebin weight must be finite and positive")
        if sed_cfg.encoder.shift_equivariant:
            first = self.timebin_weights[0]
            if any(abs(weight - first) > 1e-12 for weight in self.timebin_weights[1:]):
                raise ValueError("Shift-equivariant SEDs require equal timebin_weights")
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("tau must be finite and positive")
        if not math.isfinite(self.dead_feature_loss_weight) or self.dead_feature_loss_weight < 0:
            raise ValueError("dead_feature_loss_weight must be finite and nonnegative")
        if self.dsed_loss_weight_map is not None:
            expected = set(sed_cfg.dsed_topk_map)
            actual = set(self.dsed_loss_weight_map)
            if actual != expected:
                raise ValueError(
                    "dsed_loss_weight_map keys must exactly match dsed_topk_map keys "
                    f"({sorted(actual)} != {sorted(expected)})"
                )
            if any(
                not math.isfinite(weight) or weight < 0
                for weight in self.dsed_loss_weight_map.values()
            ):
                raise ValueError("Every SED level loss weight must be finite and nonnegative")
            if not any(weight > 0 for weight in self.dsed_loss_weight_map.values()):
                raise ValueError("At least one SED level loss weight must be positive")

    def level_weight(self, d_sed: int) -> float:
        """Return the reconstruction weight for one nested SED level."""

        if self.dsed_loss_weight_map is None:
            return 1.0
        return float(self.dsed_loss_weight_map[d_sed])


@dataclass
class TrainConfig:
    """Optimization settings for a single SED replica."""

    batch_size: int = 1024
    epochs: int = 30
    learning_rate: float = 5e-3
    use_lr_schedule: bool = True
    log_frequency: int = 100
    dead_feature_window: int = 1000
    max_dead_feature_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.log_frequency < 1 or self.dead_feature_window < 1:
            raise ValueError("logging and dead-feature windows must be positive")
        if not math.isfinite(self.max_dead_feature_fraction) or not (
            0 < self.max_dead_feature_fraction <= 1
        ):
            raise ValueError("max_dead_feature_fraction must be finite and in (0, 1]")


@dataclass
class ExperimentConfig:
    """Independent experimental replica seeds."""

    seeds: List[int] = field(default_factory=lambda: [0])

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("At least one replica seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Replica seeds must be unique")
