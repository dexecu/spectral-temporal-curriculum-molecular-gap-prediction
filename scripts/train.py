#!/usr/bin/env python3
"""Training script for Spectral-Temporal Curriculum Learning on PCQM4Mv2.

This script implements the full training pipeline with curriculum learning,
MLflow tracking, and comprehensive evaluation.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateMonitor,
    RichProgressBar, RichModelSummary
)
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
import numpy as np

from spectral_temporal_curriculum_molecular_gap_prediction.utils.config import (
    load_config, save_config, override_config_from_args, Config
)
from spectral_temporal_curriculum_molecular_gap_prediction.data.loader import PCQM4Mv2DataModule
from spectral_temporal_curriculum_molecular_gap_prediction.training.trainer import CurriculumTrainer
from spectral_temporal_curriculum_molecular_gap_prediction.evaluation.metrics import ConvergenceAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_reproducibility(seed: int) -> None:
    """Setup reproducible training environment.

    Args:
        seed: Random seed for reproducibility
    """
    # Set all random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)

    # Configure PyTorch for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    logger.info(f"Set random seed to {seed} for reproducible training")


def create_trainer(config: Config) -> pl.Trainer:
    """Create PyTorch Lightning trainer with all callbacks.

    Args:
        config: Configuration object

    Returns:
        Configured trainer
    """
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.logging.dirpath,
        filename=config.logging.filename,
        monitor=config.logging.monitor,
        mode=config.logging.mode,
        save_top_k=config.logging.save_top_k,
        save_last=config.logging.save_last,
        auto_insert_metric_name=config.logging.auto_insert_metric_name,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor=config.logging.monitor,
        mode=config.logging.mode,
        patience=config.training.early_stopping_patience,
        min_delta=config.training.early_stopping_min_delta,
        verbose=True
    )
    callbacks.append(early_stop_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Rich progress bar
    progress_bar = RichProgressBar(leave=True)
    callbacks.append(progress_bar)

    # Model summary
    model_summary = RichModelSummary(max_depth=3)
    callbacks.append(model_summary)

    # MLflow logger
    mlf_logger = MLFlowLogger(
        experiment_name=config.experiment.name,
        tracking_uri=config.experiment.tracking_uri,
        tags={str(k): v for k, v in enumerate(config.experiment.tags)}
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.device,
        devices=config.num_gpus if config.device == 'gpu' else 'auto',
        strategy=config.strategy,
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        deterministic=config.training.deterministic,
        benchmark=config.training.benchmark,
        log_every_n_steps=config.logging.log_every_n_steps,
        callbacks=callbacks,
        logger=mlf_logger,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    return trainer


def log_config_to_mlflow(config: Config) -> None:
    """Log configuration parameters to MLflow.

    Args:
        config: Configuration object
    """
    try:
        # Log all configuration parameters
        config_dict = config.to_dict()

        def log_nested_params(params: Dict[str, Any], prefix: str = '') -> None:
            """Recursively log nested parameters."""
            for key, value in params.items():
                param_name = f"{prefix}{key}" if prefix else key

                if isinstance(value, dict):
                    log_nested_params(value, f"{param_name}.")
                elif isinstance(value, (list, tuple)):
                    mlflow.log_param(param_name, str(value))
                else:
                    mlflow.log_param(param_name, value)

        log_nested_params(config_dict)

        # Log target metrics
        mlflow.log_param("target.mae_ev", config.evaluation.target_mae_ev)
        mlflow.log_param("target.convergence_speedup", 1.35)
        mlflow.log_param("target.tail_p95_mae", 0.14)

        logger.info("Configuration logged to MLflow")

    except Exception as e:
        logger.warning(f"Failed to log configuration to MLflow: {e}")


def train_model(config: Config, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
    """Train the model with curriculum learning.

    Args:
        config: Training configuration
        resume_from_checkpoint: Path to checkpoint to resume from

    Returns:
        Dictionary with training results
    """
    logger.info("Starting training pipeline")

    # Setup reproducibility
    setup_reproducibility(config.seed)

    # Initialize data module
    logger.info("Initializing data module...")
    datamodule = PCQM4Mv2DataModule(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        subset=config.data.subset,
        curriculum_strategy=config.data.curriculum_strategy,
        max_samples=config.data.max_samples,
        node_feature_dim=config.data.node_feature_dim,
        edge_feature_dim=config.data.edge_feature_dim,
        cache_spectral_features=config.data.cache_spectral_features
    )

    # Prepare data
    datamodule.prepare_data()
    datamodule.setup(stage='fit')

    # Initialize model
    logger.info("Initializing model...")
    from dataclasses import asdict
    model = CurriculumTrainer(
        model_config=asdict(config.model),
        optimizer_config=asdict(config.optimizer),
        scheduler_config=asdict(config.scheduler),
        curriculum_config=asdict(config.curriculum),
        loss_config=asdict(config.loss),
        evaluation_config=asdict(config.evaluation)
    )

    # Create trainer
    trainer = create_trainer(config)

    # Log configuration to MLflow
    with mlflow.start_run():
        log_config_to_mlflow(config)

        # Log model architecture
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("model.total_params", total_params)
        mlflow.log_param("model.trainable_params", trainable_params)

        logger.info(f"Model has {total_params:,} total parameters, {trainable_params:,} trainable")

        # Initialize convergence analyzer
        convergence_analyzer = ConvergenceAnalyzer(baseline_reference=0.11)  # Typical baseline MAE

        # Training loop with curriculum learning
        logger.info("Starting curriculum training...")

        # Use standard trainer.fit() - curriculum is handled in the model
        trainer.fit(model, datamodule, ckpt_path=resume_from_checkpoint)

        # Get final metrics
        final_train_metrics = model.metrics.compute(split='train')
        final_val_metrics = model.metrics.compute(split='val')

        # Log final metrics
        for key, value in final_train_metrics.items():
            mlflow.log_metric(f"final_train_{key}", value)
        for key, value in final_val_metrics.items():
            mlflow.log_metric(f"final_val_{key}", value)

        # Compute convergence metrics
        convergence_metrics = convergence_analyzer.compute_convergence_metrics()
        for key, value in convergence_metrics.items():
            mlflow.log_metric(f"convergence_{key}", value)

        # Save convergence plots
        try:
            plots_dir = Path("plots")
            plots_dir.mkdir(exist_ok=True)
            convergence_plot_path = plots_dir / "training_curves.png"
            convergence_analyzer.plot_training_curves(save_path=str(convergence_plot_path))
            if convergence_plot_path.exists():
                mlflow.log_artifact(str(convergence_plot_path))
        except Exception as e:
            logger.warning(f"Failed to save convergence plots: {e}")

        # Log best checkpoint as model
        if config.experiment.log_model:
            try:
                best_model_path = trainer.checkpoint_callback.best_model_path
                if best_model_path and Path(best_model_path).exists():
                    mlflow.log_artifact(best_model_path)
                    logger.info(f"Best model saved: {best_model_path}")
            except Exception as e:
                logger.warning(f"Failed to log best model: {e}")

    # Prepare results
    results = {
        'final_train_metrics': final_train_metrics,
        'final_val_metrics': final_val_metrics,
        'convergence_metrics': convergence_metrics,
        'best_model_path': trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
    }

    logger.info("Training completed successfully")
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Spectral-Temporal Curriculum Learning Model")

    # Configuration arguments
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')

    # Override arguments
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--max_epochs', type=int, help='Override max epochs')
    parser.add_argument('--hidden_dim', type=int, help='Override hidden dimension')
    parser.add_argument('--dropout', type=float, help='Override dropout rate')
    parser.add_argument('--seed', type=int, help='Override random seed')

    # Development arguments
    parser.add_argument('--dev', action='store_true',
                      help='Use development mode (subset of data)')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to use')
    parser.add_argument('--fast_dev_run', action='store_true',
                      help='Run in fast development mode')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply development overrides
    if args.dev:
        config.data.subset = True
        config.data.max_samples = 1000
        config.training.max_epochs = 5
        config.data.cache_spectral_features = False
        logger.info("Development mode enabled")

    if args.max_samples:
        config.data.max_samples = args.max_samples

    # Apply command-line overrides
    arg_overrides = {k: v for k, v in vars(args).items()
                    if v is not None and k not in ['config', 'resume', 'dev', 'max_samples', 'fast_dev_run']}

    if arg_overrides:
        config = override_config_from_args(config, arg_overrides)

    # Save effective configuration
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    effective_config_path = output_dir / "effective_config.yaml"
    save_config(config, effective_config_path)
    logger.info(f"Effective configuration saved to {effective_config_path}")

    # Run training
    try:
        results = train_model(config, resume_from_checkpoint=args.resume)

        # Print final results
        print("\n" + "="*50)
        print("TRAINING COMPLETED")
        print("="*50)

        print(f"Final Validation MAE: {results['final_val_metrics'].get('mae', 'N/A'):.4f} eV")

        target_mae = config.evaluation.target_mae_ev
        final_mae = results['final_val_metrics'].get('mae', float('inf'))
        if final_mae <= target_mae:
            print(f"✓ Target MAE achieved! ({final_mae:.4f} ≤ {target_mae:.4f} eV)")
        else:
            print(f"✗ Target MAE not achieved ({final_mae:.4f} > {target_mae:.4f} eV)")

        if 'convergence_speedup_vs_baseline' in results['convergence_metrics']:
            speedup = results['convergence_metrics']['convergence_speedup_vs_baseline']
            print(f"Convergence Speedup: {speedup:.2f}x")

        if results.get('best_model_path'):
            print(f"Best Model: {results['best_model_path']}")

        print("="*50)

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())