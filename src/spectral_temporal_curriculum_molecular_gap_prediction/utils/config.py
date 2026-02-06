"""Configuration management with validation and type checking."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from dataclasses import dataclass, field, asdict
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the SpectralTemporalNet model."""
    node_features: int = 128
    edge_features: int = 64
    hidden_dim: int = 256
    mp_layers: int = 4
    num_spectral_filters: int = 6
    max_chebyshev_order: int = 20
    fusion_type: str = 'cross_attention'
    dropout: float = 0.1
    pooling: str = 'attention'
    output_dim: int = 1


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""
    name: str = 'adamw'
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    momentum: float = 0.9  # For SGD


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""
    name: Optional[str] = 'cosine'
    T_max: int = 100
    eta_min: float = 1e-6
    gamma: float = 0.95
    factor: float = 0.5
    patience: int = 10
    total_steps: int = 10000


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    initial_fraction: float = 0.1
    final_fraction: float = 1.0
    warmup_epochs: int = 5
    total_epochs: int = 100
    growth_strategy: str = 'exponential'
    min_growth_rate: float = 0.05


@dataclass
class LossConfig:
    """Configuration for loss function."""
    base_loss: str = 'mae'
    uncertainty_weight: float = 0.1
    spectral_regularization: float = 0.01
    curriculum_weight: float = 0.05


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    target_mae_ev: float = 0.082
    convergence_window: int = 10
    tail_percentile: float = 95.0
    compute_correlations: bool = True
    track_convergence: bool = True


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_dir: str = "data"
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    subset: bool = True
    curriculum_strategy: str = "spectral_complexity"
    max_samples: Optional[int] = None
    node_feature_dim: int = 128
    edge_feature_dim: int = 64
    cache_spectral_features: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    max_epochs: int = 100
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    precision: str = '32'
    deterministic: bool = True
    benchmark: bool = False


@dataclass
class LoggingConfig:
    """Configuration for logging and checkpointing."""
    log_every_n_steps: int = 50
    save_top_k: int = 3
    monitor: str = 'val/mae'
    mode: str = 'min'
    save_last: bool = True
    dirpath: str = "checkpoints"
    filename: str = "spectral-temporal-{epoch:02d}-{val/mae:.4f}"
    auto_insert_metric_name: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    name: str = "spectral-temporal-curriculum"
    project: str = "molecular-gap-prediction"
    tags: List[str] = field(default_factory=lambda: ["spectral", "curriculum", "molecular"])
    log_model: bool = True
    log_artifacts: bool = True
    tracking_uri: str = "mlruns"


@dataclass
class Config:
    """Main configuration class containing all sub-configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Global settings
    seed: int = 42
    device: str = 'auto'
    num_gpus: int = 1
    strategy: str = 'auto'

    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_config()
        self._setup_device()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Model validation
        assert self.model.hidden_dim > 0, "Hidden dimension must be positive"
        assert self.model.mp_layers > 0, "Number of MP layers must be positive"
        assert self.model.num_spectral_filters > 0, "Number of spectral filters must be positive"
        assert self.model.fusion_type in ['concat', 'attention', 'cross_attention'], \
            "Invalid fusion type"
        assert self.model.pooling in ['mean', 'max', 'sum', 'attention'], \
            "Invalid pooling method"

        # Optimizer validation
        assert self.optimizer.name in ['adamw', 'adam', 'sgd'], "Invalid optimizer name"
        assert self.optimizer.lr > 0, "Learning rate must be positive"
        assert 0 <= self.optimizer.weight_decay <= 1, "Weight decay must be in [0, 1]"

        # Curriculum validation
        assert 0 < self.curriculum.initial_fraction <= 1, \
            "Initial curriculum fraction must be in (0, 1]"
        assert 0 < self.curriculum.final_fraction <= 1, \
            "Final curriculum fraction must be in (0, 1]"
        assert self.curriculum.initial_fraction <= self.curriculum.final_fraction, \
            "Initial fraction must be <= final fraction"
        assert self.curriculum.growth_strategy in ['linear', 'exponential', 'sigmoid'], \
            "Invalid curriculum growth strategy"

        # Data validation
        assert self.data.batch_size > 0, "Batch size must be positive"
        assert self.data.num_workers >= 0, "Number of workers must be non-negative"

        # Training validation
        assert self.training.max_epochs > 0, "Max epochs must be positive"
        assert self.training.early_stopping_patience > 0, "Early stopping patience must be positive"

    def _setup_device(self) -> None:
        """Setup device configuration."""
        if self.device == 'auto':
            if os.environ.get('CUDA_VISIBLE_DEVICES') and self.num_gpus > 0:
                self.device = 'gpu'
            else:
                self.device = 'cpu'
                self.num_gpus = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        # Handle nested configurations
        model_config = ModelConfig(**config_dict.get('model', {}))
        optimizer_config = OptimizerConfig(**config_dict.get('optimizer', {}))
        scheduler_config = SchedulerConfig(**config_dict.get('scheduler', {}))
        curriculum_config = CurriculumConfig(**config_dict.get('curriculum', {}))
        loss_config = LossConfig(**config_dict.get('loss', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        experiment_config = ExperimentConfig(**config_dict.get('experiment', {}))

        # Extract global settings
        global_keys = {'seed', 'device', 'num_gpus', 'strategy'}
        global_config = {k: v for k, v in config_dict.items() if k in global_keys}

        return cls(
            model=model_config,
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            curriculum=curriculum_config,
            loss=loss_config,
            evaluation=evaluation_config,
            data=data_config,
            training=training_config,
            logging=logging_config,
            experiment=experiment_config,
            **global_config
        )


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Loaded configuration object
    """
    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using default configuration.")
        return Config()

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            logger.warning("Empty configuration file. Using defaults.")
            return Config()

        logger.info(f"Loaded configuration from {config_path}")
        return Config.from_dict(config_dict)

    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        logger.info("Using default configuration instead.")
        return Config()


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration object to save
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        config_dict = config.to_dict()

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Saved configuration to {config_path}")

    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        raise


def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
    """Merge base configuration with override values.

    Args:
        base_config: Base configuration object
        override_config: Dictionary with override values

    Returns:
        Merged configuration object
    """
    # Convert base config to dict
    base_dict = base_config.to_dict()

    # Deep merge override config
    merged_dict = _deep_merge_dicts(base_dict, override_config)

    return Config.from_dict(merged_dict)


def _deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries.

    Args:
        dict1: Base dictionary
        dict2: Override dictionary

    Returns:
        Merged dictionary
    """
    result = deepcopy(dict1)

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def validate_config(config: Config) -> bool:
    """Validate configuration and log any issues.

    Args:
        config: Configuration to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # This will trigger validation
        config._validate_config()
        logger.info("Configuration validation passed")
        return True

    except AssertionError as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        return False


def create_default_config_file(config_path: Union[str, Path]) -> None:
    """Create a default configuration file.

    Args:
        config_path: Path where to save the default configuration
    """
    config = Config()
    save_config(config, config_path)
    logger.info(f"Created default configuration file: {config_path}")


def override_config_from_args(config: Config, args: Dict[str, Any]) -> Config:
    """Override configuration with command-line arguments.

    Args:
        config: Base configuration
        args: Command-line arguments dictionary

    Returns:
        Updated configuration
    """
    override_dict = {}

    # Map common command-line arguments to config structure
    arg_mapping = {
        'learning_rate': 'optimizer.lr',
        'batch_size': 'data.batch_size',
        'max_epochs': 'training.max_epochs',
        'hidden_dim': 'model.hidden_dim',
        'dropout': 'model.dropout',
        'seed': 'seed'
    }

    for arg_name, config_path in arg_mapping.items():
        if arg_name in args and args[arg_name] is not None:
            # Split config path and set nested value
            keys = config_path.split('.')
            current_dict = override_dict

            for key in keys[:-1]:
                if key not in current_dict:
                    current_dict[key] = {}
                current_dict = current_dict[key]

            current_dict[keys[-1]] = args[arg_name]

    if override_dict:
        config = merge_configs(config, override_dict)
        logger.info("Applied command-line argument overrides")

    return config