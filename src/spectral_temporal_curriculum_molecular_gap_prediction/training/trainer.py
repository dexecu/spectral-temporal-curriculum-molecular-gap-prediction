"""Curriculum learning trainer with spectral complexity scheduling."""

import logging
import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau,
    OneCycleLR, CyclicLR
)

from ..models.model import SpectralTemporalNet
from ..evaluation.metrics import MolecularGapMetrics
from ..data.loader import PCQM4Mv2DataModule

logger = logging.getLogger(__name__)


class CustomLoss(nn.Module):
    """Custom loss function with uncertainty weighting and regularization."""

    def __init__(
        self,
        base_loss: str = 'mae',
        uncertainty_weight: float = 0.1,
        spectral_regularization: float = 0.01,
        curriculum_weight: float = 0.05
    ):
        """Initialize custom loss function.

        Args:
            base_loss: Base loss function ('mae', 'mse', 'huber')
            uncertainty_weight: Weight for uncertainty regularization
            spectral_regularization: Weight for spectral smoothness regularization
            curriculum_weight: Weight for curriculum-aware loss adjustment
        """
        super().__init__()
        self.base_loss = base_loss
        self.uncertainty_weight = uncertainty_weight
        self.spectral_regularization = spectral_regularization
        self.curriculum_weight = curriculum_weight

        # Base loss functions
        if base_loss == 'mae':
            self.criterion = nn.L1Loss(reduction='none')
        elif base_loss == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif base_loss == 'huber':
            self.criterion = nn.SmoothL1Loss(reduction='none', beta=1.0)
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        batch_data: Optional[Any] = None,
        model: Optional[nn.Module] = None,
        curriculum_difficulty: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute custom loss with regularization terms.

        Args:
            predictions: Model predictions [batch_size, 1]
            targets: Ground truth targets [batch_size, 1]
            batch_data: Batch data for accessing graph properties
            model: Model for computing regularization terms
            curriculum_difficulty: Difficulty scores for curriculum weighting

        Returns:
            Dictionary with loss components
        """
        # Base prediction loss
        base_loss = self.criterion(predictions, targets)

        # Curriculum weighting (easier samples get higher weights initially)
        if curriculum_difficulty is not None and self.curriculum_weight > 0:
            # Inverse difficulty weighting (easier samples weighted more)
            curriculum_weights = 1.0 - curriculum_difficulty.clamp(0, 1)
            curriculum_weights = curriculum_weights / curriculum_weights.sum()
            base_loss = base_loss * curriculum_weights.view(-1, 1)

        # Aggregate base loss
        base_loss_mean = base_loss.mean()

        loss_dict = {'base_loss': base_loss_mean}

        # Spectral regularization (encourage smooth spectral filters)
        if model is not None and self.spectral_regularization > 0:
            spectral_reg = self._compute_spectral_regularization(model)
            loss_dict['spectral_reg'] = spectral_reg
        else:
            loss_dict['spectral_reg'] = torch.tensor(0.0, device=predictions.device)

        # Total loss
        total_loss = (
            base_loss_mean +
            self.spectral_regularization * loss_dict['spectral_reg']
        )

        loss_dict['total_loss'] = total_loss

        return loss_dict

    def _compute_spectral_regularization(self, model: nn.Module) -> torch.Tensor:
        """Compute spectral regularization term.

        Encourages smoothness in spectral filter coefficients.
        """
        reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, module in model.named_modules():
            if hasattr(module, 'weight') and 'spectral' in name.lower():
                # L2 regularization on spectral filter weights
                reg_loss = reg_loss + torch.norm(module.weight, p=2)

        return reg_loss


class CurriculumScheduler(ABC):
    """Abstract base class for curriculum learning schedulers."""

    def __init__(self):
        self.current_step = 0
        self.current_epoch = 0

    @abstractmethod
    def get_difficulty_fraction(self, step: int, epoch: int) -> float:
        """Get the fraction of training data to include based on difficulty.

        Args:
            step: Current training step
            epoch: Current epoch

        Returns:
            Fraction of data to include (0.0 to 1.0)
        """
        pass

    def step(self) -> None:
        """Update scheduler state."""
        self.current_step += 1

    def epoch(self) -> None:
        """Update epoch state."""
        self.current_epoch += 1
        self.current_step = 0


class SpectralComplexityScheduler(CurriculumScheduler):
    """Curriculum scheduler based on spectral complexity of molecular graphs."""

    def __init__(
        self,
        initial_fraction: float = 0.1,
        final_fraction: float = 1.0,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        growth_strategy: str = 'exponential',
        min_growth_rate: float = 0.05
    ):
        """Initialize spectral complexity scheduler.

        Args:
            initial_fraction: Starting fraction of easiest data
            final_fraction: Final fraction of data (usually 1.0)
            warmup_epochs: Number of epochs for initial warmup
            total_epochs: Total training epochs
            growth_strategy: How to grow difficulty ('linear', 'exponential', 'sigmoid')
            min_growth_rate: Minimum growth rate per epoch
        """
        super().__init__()
        self.initial_fraction = initial_fraction
        self.final_fraction = final_fraction
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.growth_strategy = growth_strategy
        self.min_growth_rate = min_growth_rate

    def get_difficulty_fraction(self, step: int, epoch: int) -> float:
        """Get difficulty fraction based on spectral complexity curriculum."""
        # Warmup period: use initial fraction
        if epoch < self.warmup_epochs:
            return self.initial_fraction

        # Progress through curriculum
        progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, progress)

        if self.growth_strategy == 'linear':
            fraction = self.initial_fraction + progress * (self.final_fraction - self.initial_fraction)

        elif self.growth_strategy == 'exponential':
            # Exponential growth: slower at first, then faster
            exp_progress = math.exp(3 * progress) - 1
            exp_progress = exp_progress / (math.exp(3) - 1)
            fraction = self.initial_fraction + exp_progress * (self.final_fraction - self.initial_fraction)

        elif self.growth_strategy == 'sigmoid':
            # Sigmoid growth: slow-fast-slow
            sigmoid_progress = 1 / (1 + math.exp(-10 * (progress - 0.5)))
            fraction = self.initial_fraction + sigmoid_progress * (self.final_fraction - self.initial_fraction)

        else:
            # Default to linear
            fraction = self.initial_fraction + progress * (self.final_fraction - self.initial_fraction)

        # Ensure minimum growth
        min_fraction_at_epoch = self.initial_fraction + epoch * self.min_growth_rate
        fraction = max(fraction, min_fraction_at_epoch)

        return min(fraction, self.final_fraction)


class CurriculumTrainer(pl.LightningModule):
    """PyTorch Lightning trainer with spectral complexity curriculum learning."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any],
        curriculum_config: Dict[str, Any],
        loss_config: Dict[str, Any],
        evaluation_config: Dict[str, Any]
    ):
        """Initialize curriculum trainer.

        Args:
            model_config: Model configuration parameters
            optimizer_config: Optimizer configuration
            scheduler_config: Learning rate scheduler configuration
            curriculum_config: Curriculum learning configuration
            loss_config: Loss function configuration
            evaluation_config: Evaluation metrics configuration
        """
        super().__init__()
        self.save_hyperparameters()

        # Initialize model
        self.model = SpectralTemporalNet(**model_config)

        # Initialize loss function
        self.loss_fn = CustomLoss(**loss_config)

        # Initialize curriculum scheduler
        self.curriculum_scheduler = SpectralComplexityScheduler(**curriculum_config)

        # Initialize evaluation metrics
        self.metrics = MolecularGapMetrics(**evaluation_config)

        # Store configurations
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        # Training state
        self.current_curriculum_fraction = curriculum_config.get('initial_fraction', 0.1)

        # Automatic optimization
        self.automatic_optimization = True

    def forward(self, batch) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(batch)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Training step with curriculum learning.

        Args:
            batch: Training batch
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        predictions = self.forward(batch)
        targets = batch.y.view(-1, 1)

        # Get curriculum difficulty scores if available
        curriculum_difficulty = None
        if hasattr(batch, 'spectral_complexity'):
            curriculum_difficulty = batch.spectral_complexity

        # Compute loss
        loss_dict = self.loss_fn(
            predictions=predictions,
            targets=targets,
            batch_data=batch,
            model=self.model,
            curriculum_difficulty=curriculum_difficulty
        )

        # Log losses
        self.log('train/loss', loss_dict['total_loss'], prog_bar=True)
        self.log('train/base_loss', loss_dict['base_loss'])
        if 'spectral_reg' in loss_dict:
            self.log('train/spectral_reg', loss_dict['spectral_reg'])

        # Update metrics
        self.metrics.update(predictions, targets, split='train')

        return loss_dict['total_loss']

    def validation_step(self, batch, batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Validation batch
            batch_idx: Batch index
        """
        predictions = self.forward(batch)
        targets = batch.y.view(-1, 1)

        # Compute loss (without curriculum weighting for fair comparison)
        loss_dict = self.loss_fn(
            predictions=predictions,
            targets=targets,
            batch_data=batch,
            model=self.model,
            curriculum_difficulty=None  # No curriculum weighting in validation
        )

        # Log validation loss
        self.log('val/loss', loss_dict['total_loss'], prog_bar=True)
        self.log('val/base_loss', loss_dict['base_loss'])

        # Update metrics
        self.metrics.update(predictions, targets, split='val')

    def test_step(self, batch, batch_idx: int) -> None:
        """Test step.

        Args:
            batch: Test batch
            batch_idx: Batch index
        """
        predictions = self.forward(batch)
        targets = batch.y.view(-1, 1)

        # Compute loss
        loss_dict = self.loss_fn(
            predictions=predictions,
            targets=targets,
            batch_data=batch,
            model=self.model,
            curriculum_difficulty=None
        )

        # Log test loss
        self.log('test/loss', loss_dict['total_loss'])
        self.log('test/base_loss', loss_dict['base_loss'])

        # Update metrics
        self.metrics.update(predictions, targets, split='test')

    def on_train_epoch_start(self) -> None:
        """Called at the start of each training epoch."""
        # Update curriculum scheduler
        self.curriculum_scheduler.epoch()

        # Get current difficulty fraction
        self.current_curriculum_fraction = self.curriculum_scheduler.get_difficulty_fraction(
            step=0, epoch=self.current_epoch
        )

        # Log curriculum progress
        self.log('curriculum/fraction', self.current_curriculum_fraction)

        logger.info(
            f"Epoch {self.current_epoch}: "
            f"Using {self.current_curriculum_fraction:.2%} of training data"
        )

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        # Compute and log training metrics
        train_metrics = self.metrics.compute(split='train')
        for key, value in train_metrics.items():
            self.log(f'train/{key}', value)

        # Reset training metrics
        self.metrics.reset(split='train')

    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch."""
        # Compute and log validation metrics
        val_metrics = self.metrics.compute(split='val')
        for key, value in val_metrics.items():
            self.log(f'val/{key}', value)

        # Reset validation metrics
        self.metrics.reset(split='val')

    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        # Compute and log test metrics
        test_metrics = self.metrics.compute(split='test')
        for key, value in test_metrics.items():
            self.log(f'test/{key}', value)

        # Reset test metrics
        self.metrics.reset(split='test')

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        # Initialize optimizer
        if self.optimizer_config['name'] == 'adamw':
            optimizer = AdamW(
                self.parameters(),
                lr=self.optimizer_config['lr'],
                weight_decay=self.optimizer_config.get('weight_decay', 0.01),
                betas=self.optimizer_config.get('betas', (0.9, 0.999))
            )
        elif self.optimizer_config['name'] == 'adam':
            optimizer = Adam(
                self.parameters(),
                lr=self.optimizer_config['lr'],
                weight_decay=self.optimizer_config.get('weight_decay', 0.01),
                betas=self.optimizer_config.get('betas', (0.9, 0.999))
            )
        elif self.optimizer_config['name'] == 'sgd':
            optimizer = SGD(
                self.parameters(),
                lr=self.optimizer_config['lr'],
                momentum=self.optimizer_config.get('momentum', 0.9),
                weight_decay=self.optimizer_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_config['name']}")

        config = {"optimizer": optimizer}

        # Add scheduler if specified
        if self.scheduler_config.get('name') is not None:
            if self.scheduler_config['name'] == 'cosine':
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=self.scheduler_config.get('T_max', 100),
                    eta_min=self.scheduler_config.get('eta_min', 0.0)
                )
                config["lr_scheduler"] = {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }

            elif self.scheduler_config['name'] == 'exponential':
                scheduler = ExponentialLR(
                    optimizer,
                    gamma=self.scheduler_config.get('gamma', 0.95)
                )
                config["lr_scheduler"] = {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }

            elif self.scheduler_config['name'] == 'plateau':
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=self.scheduler_config.get('factor', 0.5),
                    patience=self.scheduler_config.get('patience', 10),
                    verbose=True
                )
                config["lr_scheduler"] = {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1
                }

            elif self.scheduler_config['name'] == 'onecycle':
                # Get total training steps
                total_steps = self.scheduler_config.get('total_steps', 10000)

                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=self.optimizer_config['lr'],
                    total_steps=total_steps,
                    anneal_strategy='cos'
                )
                config["lr_scheduler"] = {
                    "scheduler": scheduler,
                    "interval": "step"
                }

        return config

    def get_curriculum_dataloader(
        self,
        datamodule: PCQM4Mv2DataModule,
        stage: str = 'train'
    ):
        """Get curriculum-based data loader.

        Args:
            datamodule: Data module with curriculum support
            stage: Training stage ('train', 'val', 'test')

        Returns:
            Data loader with curriculum subset
        """
        if stage != 'train':
            # No curriculum for validation/test
            if stage == 'val':
                return datamodule.val_dataloader()
            else:
                return datamodule.test_dataloader()

        # Get curriculum data loader for training
        return datamodule.get_curriculum_dataloader(
            stage='train',
            difficulty_fraction=self.current_curriculum_fraction,
            shuffle=True
        )

    def predict_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step for inference.

        Args:
            batch: Input batch
            batch_idx: Batch index

        Returns:
            Dictionary with predictions and features
        """
        predictions = self.forward(batch)

        # Get attention weights for interpretability
        attention_weights = self.model.get_attention_weights(batch)

        return {
            'predictions': predictions,
            'targets': batch.y.view(-1, 1) if hasattr(batch, 'y') else None,
            'attention_weights': attention_weights
        }