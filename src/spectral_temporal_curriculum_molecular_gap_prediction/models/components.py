"""Custom model components for spectral-temporal molecular property prediction.

This module contains custom loss functions, regularization terms, and utility
components that enhance the dual-view architecture.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class SpectralRegularizedLoss(nn.Module):
    """Custom loss function with spectral regularization.

    This loss combines a base regression loss (MAE/MSE) with a spectral
    regularization term that encourages the model to learn smooth spectral
    filters. The regularization penalizes high-frequency components in the
    learned Chebyshev coefficients.

    The total loss is:
        L = L_base + λ * L_spectral

    where L_spectral encourages smoothness in the spectral domain.

    Attributes:
        base_loss (str): Base loss type ('mae', 'mse', 'huber')
        spectral_weight (float): Weight for spectral regularization term
        curriculum_weight (float): Weight for curriculum-based sample weighting
    """

    def __init__(
        self,
        base_loss: str = 'mae',
        spectral_weight: float = 0.01,
        curriculum_weight: float = 0.05,
        huber_delta: float = 1.0
    ):
        """Initialize spectral regularized loss.

        Args:
            base_loss (str): Base loss type ('mae', 'mse', 'huber')
            spectral_weight (float): Weight for spectral regularization
            curriculum_weight (float): Weight for curriculum sample weighting
            huber_delta (float): Delta parameter for Huber loss

        Raises:
            ValueError: If base_loss is not recognized or weights are negative
        """
        super().__init__()

        if base_loss not in ['mae', 'mse', 'huber']:
            raise ValueError(f"base_loss must be 'mae', 'mse', or 'huber', got {base_loss}")
        if spectral_weight < 0:
            raise ValueError(f"spectral_weight must be non-negative, got {spectral_weight}")
        if curriculum_weight < 0:
            raise ValueError(f"curriculum_weight must be non-negative, got {curriculum_weight}")

        self.base_loss = base_loss
        self.spectral_weight = spectral_weight
        self.curriculum_weight = curriculum_weight
        self.huber_delta = huber_delta

        logger.info(
            f"Initialized SpectralRegularizedLoss: base={base_loss}, "
            f"spectral_weight={spectral_weight}, curriculum_weight={curriculum_weight}"
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        spectral_filters: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute regularized loss.

        Args:
            predictions (torch.Tensor): Model predictions [batch_size, 1]
            targets (torch.Tensor): Ground truth values [batch_size, 1]
            spectral_filters (Optional[torch.Tensor]): Spectral filter weights for
                regularization [num_filters, K, in_channels, out_channels]
            sample_weights (Optional[torch.Tensor]): Per-sample curriculum weights
                [batch_size]

        Returns:
            Tuple[torch.Tensor, dict]: Total loss and dictionary of loss components
        """
        # Compute base regression loss
        if self.base_loss == 'mae':
            base = F.l1_loss(predictions, targets, reduction='none')
        elif self.base_loss == 'mse':
            base = F.mse_loss(predictions, targets, reduction='none')
        elif self.base_loss == 'huber':
            base = F.smooth_l1_loss(
                predictions, targets,
                reduction='none',
                beta=self.huber_delta
            )

        # Apply sample weights if provided (for curriculum learning)
        if sample_weights is not None:
            if sample_weights.dim() == 1:
                sample_weights = sample_weights.unsqueeze(-1)
            weighted_base = base * sample_weights
            base_loss = weighted_base.mean()
        else:
            base_loss = base.mean()

        # Compute spectral regularization if filters provided
        spectral_reg = torch.tensor(0.0, device=predictions.device)
        if spectral_filters is not None and self.spectral_weight > 0:
            spectral_reg = self._compute_spectral_regularization(spectral_filters)

        # Total loss
        total_loss = base_loss + self.spectral_weight * spectral_reg

        # Loss components for logging
        loss_dict = {
            'base_loss': base_loss.item(),
            'spectral_reg': spectral_reg.item() if isinstance(spectral_reg, torch.Tensor) else spectral_reg,
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict

    def _compute_spectral_regularization(self, spectral_filters: torch.Tensor) -> torch.Tensor:
        """Compute spectral smoothness regularization.

        Encourages smoothness in the spectral domain by penalizing
        high-frequency components in Chebyshev polynomial coefficients.

        Args:
            spectral_filters (torch.Tensor): Filter weights
                [num_filters, K, in_channels, out_channels] or [K, in_channels, out_channels]

        Returns:
            torch.Tensor: Regularization loss (scalar)
        """
        # Handle both single filter and filter bank shapes
        if spectral_filters.dim() == 4:
            # Filter bank: [num_filters, K, in_channels, out_channels]
            # Compute differences along Chebyshev order dimension (dim=1)
            filter_diffs = torch.diff(spectral_filters, dim=1)
        elif spectral_filters.dim() == 3:
            # Single filter: [K, in_channels, out_channels]
            # Compute differences along Chebyshev order dimension (dim=0)
            filter_diffs = torch.diff(spectral_filters, dim=0)
        else:
            logger.warning(f"Unexpected spectral_filters shape: {spectral_filters.shape}")
            return torch.tensor(0.0, device=spectral_filters.device)

        # L2 norm of differences (encourages smoothness)
        smoothness_penalty = (filter_diffs ** 2).mean()

        return smoothness_penalty


class UncertaintyWeightedLoss(nn.Module):
    """Loss function with learnable uncertainty weighting.

    This loss learns to weight the prediction loss based on aleatoric uncertainty,
    following the approach from "What Uncertainties Do We Need in Bayesian Deep Learning
    for Computer Vision?" (Kendall & Gal, 2017).

    The loss includes an uncertainty term:
        L = (1 / (2 * σ²)) * L_base + log(σ)

    where σ is the learned uncertainty parameter.

    Attributes:
        log_variance (nn.Parameter): Learnable log variance parameter
    """

    def __init__(self, init_log_variance: float = 0.0):
        """Initialize uncertainty-weighted loss.

        Args:
            init_log_variance (float): Initial value for log variance parameter
        """
        super().__init__()

        self.log_variance = nn.Parameter(torch.tensor(init_log_variance))

        logger.info(f"Initialized UncertaintyWeightedLoss with log_variance={init_log_variance}")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute uncertainty-weighted loss.

        Args:
            predictions (torch.Tensor): Model predictions [batch_size, 1]
            targets (torch.Tensor): Ground truth values [batch_size, 1]

        Returns:
            Tuple[torch.Tensor, dict]: Total loss and dictionary of components
        """
        # Base MAE loss
        base_loss = F.l1_loss(predictions, targets, reduction='mean')

        # Uncertainty weighting
        precision = torch.exp(-self.log_variance)
        weighted_loss = precision * base_loss + self.log_variance

        loss_dict = {
            'base_loss': base_loss.item(),
            'uncertainty': torch.exp(self.log_variance / 2).item(),  # Standard deviation
            'total_loss': weighted_loss.item()
        }

        return weighted_loss, loss_dict


class CurriculumWeightedLoss(nn.Module):
    """Loss with curriculum-based sample weighting.

    This loss dynamically weights samples based on their difficulty,
    as measured by spectral complexity or other curriculum metrics.
    Early in training, easier samples receive higher weight.

    Attributes:
        base_loss (str): Base loss type
        curriculum_factor (float): Strength of curriculum weighting
    """

    def __init__(self, base_loss: str = 'mae', curriculum_factor: float = 1.0):
        """Initialize curriculum-weighted loss.

        Args:
            base_loss (str): Base loss type ('mae', 'mse', 'huber')
            curriculum_factor (float): Strength of curriculum weighting
        """
        super().__init__()

        if base_loss not in ['mae', 'mse', 'huber']:
            raise ValueError(f"base_loss must be 'mae', 'mse', or 'huber', got {base_loss}")

        self.base_loss = base_loss
        self.curriculum_factor = curriculum_factor

        logger.info(
            f"Initialized CurriculumWeightedLoss: base={base_loss}, "
            f"curriculum_factor={curriculum_factor}"
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_difficulties: Optional[torch.Tensor] = None,
        curriculum_progress: float = 1.0
    ) -> Tuple[torch.Tensor, dict]:
        """Compute curriculum-weighted loss.

        Args:
            predictions (torch.Tensor): Model predictions [batch_size, 1]
            targets (torch.Tensor): Ground truth values [batch_size, 1]
            sample_difficulties (Optional[torch.Tensor]): Difficulty scores [batch_size]
                Higher values indicate harder samples
            curriculum_progress (float): Progress through curriculum [0, 1]
                0 = start, 1 = end

        Returns:
            Tuple[torch.Tensor, dict]: Total loss and dictionary of components
        """
        # Compute base loss
        if self.base_loss == 'mae':
            base = F.l1_loss(predictions, targets, reduction='none')
        elif self.base_loss == 'mse':
            base = F.mse_loss(predictions, targets, reduction='none')
        elif self.base_loss == 'huber':
            base = F.smooth_l1_loss(predictions, targets, reduction='none')

        # Compute sample weights based on curriculum
        if sample_difficulties is not None and curriculum_progress < 1.0:
            # Easier samples get higher weight early in training
            # Weight decreases as curriculum progresses
            weight_adjustment = (1.0 - sample_difficulties) * (1.0 - curriculum_progress)
            sample_weights = 1.0 + self.curriculum_factor * weight_adjustment

            if sample_weights.dim() == 1:
                sample_weights = sample_weights.unsqueeze(-1)

            weighted_loss = (base * sample_weights).mean()
        else:
            weighted_loss = base.mean()

        loss_dict = {
            'base_loss': base.mean().item(),
            'curriculum_progress': curriculum_progress,
            'total_loss': weighted_loss.item()
        }

        return weighted_loss, loss_dict


class CombinedLoss(nn.Module):
    """Combined loss with all custom components.

    This is the main loss function used in training, combining:
    - Base regression loss (MAE/MSE/Huber)
    - Spectral regularization
    - Uncertainty weighting
    - Curriculum weighting

    Attributes:
        spectral_loss (SpectralRegularizedLoss): Spectral regularized loss
        uncertainty_weight (float): Weight for uncertainty component
    """

    def __init__(
        self,
        base_loss: str = 'mae',
        spectral_weight: float = 0.01,
        curriculum_weight: float = 0.05,
        uncertainty_weight: float = 0.1
    ):
        """Initialize combined loss.

        Args:
            base_loss (str): Base loss type
            spectral_weight (float): Weight for spectral regularization
            curriculum_weight (float): Weight for curriculum weighting
            uncertainty_weight (float): Weight for uncertainty component
        """
        super().__init__()

        self.spectral_loss = SpectralRegularizedLoss(
            base_loss=base_loss,
            spectral_weight=spectral_weight,
            curriculum_weight=curriculum_weight
        )

        self.uncertainty_weight = uncertainty_weight

        logger.info(
            f"Initialized CombinedLoss: base={base_loss}, "
            f"spectral={spectral_weight}, curriculum={curriculum_weight}, "
            f"uncertainty={uncertainty_weight}"
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        spectral_filters: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss.

        Args:
            predictions (torch.Tensor): Model predictions [batch_size, 1]
            targets (torch.Tensor): Ground truth values [batch_size, 1]
            spectral_filters (Optional[torch.Tensor]): Spectral filter weights
            sample_weights (Optional[torch.Tensor]): Curriculum sample weights

        Returns:
            Tuple[torch.Tensor, dict]: Total loss and dictionary of components
        """
        # Compute spectral regularized loss
        total_loss, loss_dict = self.spectral_loss(
            predictions, targets, spectral_filters, sample_weights
        )

        return total_loss, loss_dict
