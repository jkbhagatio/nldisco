"""Hydra schemas shared by one-off training and sweep entry points."""

from dataclasses import dataclass, field
from typing import Any, Optional

from beartype import beartype
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from nldisco.config import DecoderConfig, EncoderConfig, LossConfig, TrainConfig
from nldisco.evaluation import EvaluationConfig


@dataclass
class EncoderSettings(EncoderConfig):
    """OmegaConf-compatible encoder schema; runtime config validates choices."""

    type: str = "FlatWindow"


@dataclass
class DecoderSettings(DecoderConfig):
    """OmegaConf-compatible decoder schema."""

    output_activation: str = "relu"
    temporal_alignment: str = "causal"


@dataclass
class ModelSettings:
    """Architecture; input and output unit counts are inferred from data."""

    seq_len: int = 8
    dsed_topk_map: dict[int, int] = field(default_factory=dict)
    encoder: EncoderSettings = field(default_factory=EncoderSettings)
    decoder: DecoderSettings = field(default_factory=DecoderSettings)
    inference_sparsity: str = "training_threshold"
    threshold_decay: float = 0.99
    dtype: str = "float32"


@dataclass
class LossSettings(LossConfig):
    """Empty weights mean uniform weights across the configured window."""

    timebin_weights: list[float] = field(default_factory=list)
    type: str = "msle"


@dataclass
class DataSettings:
    """NumPy files with timebin-by-unit activity and optional row metadata."""

    path: str = MISSING
    normalization: str = "none"
    valid_rows: Optional[str] = None
    trial_ids: Optional[str] = None
    session_ids: Optional[str] = None
    timestamps: Optional[str] = None
    expected_bin_size: Optional[float] = None
    split: str = "chronological"
    train_fraction: float = 0.8
    train_sessions: list[str] = field(default_factory=list)
    split_seed: int = 0
    stride: Optional[int] = None
    target_path: Optional[str] = None
    target_normalization: Optional[str] = None
    target_valid_rows: Optional[str] = None
    target_trial_ids: Optional[str] = None
    target_session_ids: Optional[str] = None
    target_timestamps: Optional[str] = None
    input_population: Optional[str] = None
    target_population: Optional[str] = None


DATA_PATH_FIELDS = (
    "path", "valid_rows", "trial_ids", "session_ids", "timestamps",
    "target_path", "target_valid_rows", "target_trial_ids", "target_session_ids",
    "target_timestamps",
)


@dataclass
class SlurmSettings:
    """Resources per trial (or per W&B agent), on one Slurm node."""

    partition: Optional[str] = None
    account: Optional[str] = None
    constraint: Optional[str] = None
    timeout_min: int = 60
    cpus_per_task: int = 4
    mem_gb: float = 16.0
    gpus_per_node: int = 0


@dataclass
class ExecutionSettings:
    """Local process workers or Submitit Slurm jobs."""

    backend: str = "local"
    devices: list[str] = field(default_factory=lambda: ["cpu"])
    max_parallel: int = 1
    threads_per_worker: int = 1
    wait: bool = True
    slurm: SlurmSettings = field(default_factory=SlurmSettings)


@dataclass
class WandbSettings:
    """Credentials come from the environment, never from application config."""

    enabled: bool = False
    project: str = "nldisco"
    entity: Optional[str] = None
    group: Optional[str] = None
    mode: str = "online"
    sweep_id: Optional[str] = None


@dataclass
class SearchSettings:
    """Dotted parameter paths with W&B-style values or distribution specs."""

    backend: str = "local"
    method: str = "grid"
    max_runs: int = 10
    seed: int = 0
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """Application schema, also usable through Hydra's compose API."""

    mode: str = "train"
    seed: int = 0
    output_dir: str = "outputs/nldisco"
    env_file: Optional[str] = ".env"
    data: DataSettings = field(default_factory=DataSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    loss: LossSettings = field(default_factory=LossSettings)
    training: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    wandb: WandbSettings = field(default_factory=WandbSettings)
    search: SearchSettings = field(default_factory=SearchSettings)


@beartype
def register_configs() -> None:
    """Register the shared schema for packaged and external YAML configs."""
    ConfigStore.instance().store(name="nldisco_schema", node=RunConfig)
