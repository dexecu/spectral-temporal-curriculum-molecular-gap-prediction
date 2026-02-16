"""Evaluation metrics for molecular property prediction with statistical analysis."""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MolecularGapMetrics:
    """Comprehensive metrics for molecular HOMO-LUMO gap prediction."""

    def __init__(
        self,
        target_mae_ev: float = 0.082,
        convergence_window: int = 10,
        tail_percentile: float = 95.0,
        compute_correlations: bool = True,
        track_convergence: bool = True
    ):
        """Initialize molecular gap metrics.

        Args:
            target_mae_ev: Target MAE in eV for tracking performance
            convergence_window: Window size for convergence tracking
            tail_percentile: Percentile for tail error analysis
            compute_correlations: Whether to compute correlation metrics
            track_convergence: Whether to track convergence statistics
        """
        self.target_mae_ev = target_mae_ev
        self.convergence_window = convergence_window
        self.tail_percentile = tail_percentile
        self.compute_correlations = compute_correlations
        self.track_convergence = track_convergence

        # Initialize metric collections for each split
        self.metrics = {
            'train': self._create_metric_collection(),
            'val': self._create_metric_collection(),
            'test': self._create_metric_collection()
        }

        # Storage for predictions and targets
        self.predictions = defaultdict(list)
        self.targets = defaultdict(list)

        # Convergence tracking
        if self.track_convergence:
            self.loss_history = defaultdict(lambda: deque(maxlen=convergence_window))
            self.metric_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=convergence_window)))

    def _create_metric_collection(self) -> MetricCollection:
        """Create a collection of standard regression metrics."""
        return MetricCollection({
            'mae': MeanAbsoluteError(),
            'mse': MeanSquaredError(),
            'rmse': MeanSquaredError(squared=False),
        })

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        split: str = 'train'
    ) -> None:
        """Update metrics with new predictions and targets.

        Args:
            predictions: Model predictions [batch_size, 1]
            targets: Ground truth targets [batch_size, 1]
            split: Data split ('train', 'val', 'test')
        """
        if split not in self.metrics:
            logger.warning(f"Unknown split: {split}")
            return

        # Ensure tensors are on CPU for metric computation
        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()

        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Update standard metrics
        self.metrics[split].update(predictions, targets)

        # Store for additional analysis
        self.predictions[split].extend(predictions.numpy().tolist())
        self.targets[split].extend(targets.numpy().tolist())

    def compute(self, split: str = 'train') -> Dict[str, float]:
        """Compute all metrics for a given split.

        Args:
            split: Data split to compute metrics for

        Returns:
            Dictionary of computed metrics
        """
        if split not in self.metrics:
            logger.warning(f"Unknown split: {split}")
            return {}

        # Compute standard metrics
        standard_metrics = self.metrics[split].compute()

        # Convert to Python floats
        result = {k: float(v) for k, v in standard_metrics.items()}

        # Compute additional molecular-specific metrics
        if len(self.predictions[split]) > 0:
            pred_array = np.array(self.predictions[split])
            target_array = np.array(self.targets[split])

            # Compute additional metrics
            additional_metrics = self._compute_additional_metrics(pred_array, target_array)
            result.update(additional_metrics)

            # Update convergence tracking
            if self.track_convergence:
                self._update_convergence_metrics(result, split)

        return result

    def _compute_additional_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute additional molecular-specific metrics."""
        metrics = {}

        # Absolute errors for detailed analysis
        abs_errors = np.abs(predictions - targets)

        # Percentile errors (important for tail behavior)
        metrics['mae_p50'] = float(np.percentile(abs_errors, 50))
        metrics['mae_p90'] = float(np.percentile(abs_errors, 90))
        metrics[f'mae_p{int(self.tail_percentile)}'] = float(np.percentile(abs_errors, self.tail_percentile))
        metrics['mae_p99'] = float(np.percentile(abs_errors, 99))

        # Error distribution statistics
        metrics['mae_std'] = float(np.std(abs_errors))
        metrics['mae_max'] = float(np.max(abs_errors))

        # Fraction of predictions within target accuracy
        within_target = np.mean(abs_errors <= self.target_mae_ev)
        metrics['fraction_within_target'] = float(within_target)

        # Large error analysis (errors > 2 * target)
        large_error_threshold = 2 * self.target_mae_ev
        large_errors = abs_errors > large_error_threshold
        metrics['fraction_large_errors'] = float(np.mean(large_errors))

        if np.any(large_errors):
            metrics['mae_large_errors'] = float(np.mean(abs_errors[large_errors]))
        else:
            metrics['mae_large_errors'] = 0.0

        # Relative errors (for non-zero targets)
        nonzero_mask = np.abs(targets) > 1e-6
        if np.any(nonzero_mask):
            rel_errors = np.abs((predictions[nonzero_mask] - targets[nonzero_mask]) / targets[nonzero_mask])
            metrics['mape'] = float(np.mean(rel_errors))  # Mean Absolute Percentage Error
            metrics['mape_p90'] = float(np.percentile(rel_errors, 90))

        # Bias analysis
        bias = predictions - targets
        metrics['bias'] = float(np.mean(bias))
        metrics['bias_abs'] = float(np.mean(np.abs(bias)))

        # Correlation analysis
        if self.compute_correlations and len(predictions) > 1:
            try:
                pearson_r, pearson_p = pearsonr(predictions, targets)
                spearman_r, spearman_p = spearmanr(predictions, targets)

                metrics['pearson_r'] = float(pearson_r)
                metrics['pearson_p'] = float(pearson_p)
                metrics['spearman_r'] = float(spearman_r)
                metrics['spearman_p'] = float(spearman_p)

                # R-squared
                metrics['r2'] = float(pearson_r ** 2)

            except Exception as e:
                logger.warning(f"Failed to compute correlations: {e}")

        return metrics

    def _update_convergence_metrics(self, metrics: Dict[str, float], split: str) -> None:
        """Update convergence tracking metrics."""
        # Track MAE convergence
        mae = metrics.get('mae', float('inf'))
        self.metric_history[split]['mae'].append(mae)

        # Compute convergence statistics if we have enough history
        if len(self.metric_history[split]['mae']) >= self.convergence_window:
            mae_history = list(self.metric_history[split]['mae'])

            # Convergence rate (improvement per step)
            if len(mae_history) > 1:
                improvements = [-1 * (mae_history[i] - mae_history[i-1]) for i in range(1, len(mae_history))]
                metrics['convergence_rate'] = float(np.mean(improvements))
                metrics['convergence_stability'] = float(np.std(improvements))

            # Check if converged (small changes in recent history)
            recent_var = np.var(mae_history[-min(5, len(mae_history)):])
            metrics['convergence_variance'] = float(recent_var)
            metrics['is_converged'] = float(recent_var < 1e-6)

    def reset(self, split: Optional[str] = None) -> None:
        """Reset metrics for a given split or all splits.

        Args:
            split: Split to reset, or None to reset all splits
        """
        if split is not None:
            if split in self.metrics:
                self.metrics[split].reset()
                self.predictions[split].clear()
                self.targets[split].clear()
        else:
            # Reset all splits
            for s in self.metrics:
                self.metrics[s].reset()
                self.predictions[s].clear()
                self.targets[s].clear()

    def get_error_analysis(self, split: str = 'test') -> Dict[str, Any]:
        """Perform detailed error analysis.

        Args:
            split: Data split to analyze

        Returns:
            Dictionary with error analysis results
        """
        if len(self.predictions[split]) == 0:
            logger.warning(f"No data available for split: {split}")
            return {}

        pred_array = np.array(self.predictions[split])
        target_array = np.array(self.targets[split])
        errors = pred_array - target_array
        abs_errors = np.abs(errors)

        analysis = {
            'error_distribution': {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'skewness': float(stats.skew(errors)),
                'kurtosis': float(stats.kurtosis(errors)),
                'min': float(np.min(errors)),
                'max': float(np.max(errors))
            },
            'abs_error_distribution': {
                'mean': float(np.mean(abs_errors)),
                'std': float(np.std(abs_errors)),
                'median': float(np.median(abs_errors)),
                'q25': float(np.percentile(abs_errors, 25)),
                'q75': float(np.percentile(abs_errors, 75)),
                'q90': float(np.percentile(abs_errors, 90)),
                'q95': float(np.percentile(abs_errors, 95)),
                'q99': float(np.percentile(abs_errors, 99))
            }
        }

        # Outlier analysis
        q75, q25 = np.percentile(abs_errors, [75, 25])
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr
        outliers = abs_errors > outlier_threshold

        analysis['outliers'] = {
            'threshold': float(outlier_threshold),
            'count': int(np.sum(outliers)),
            'fraction': float(np.mean(outliers)),
            'mean_error': float(np.mean(abs_errors[outliers])) if np.any(outliers) else 0.0
        }

        return analysis

    def create_error_plots(self, split: str = 'test', save_path: Optional[str] = None) -> None:
        """Create diagnostic plots for error analysis.

        Args:
            split: Data split to plot
            save_path: Optional path to save plots
        """
        if len(self.predictions[split]) == 0:
            logger.warning(f"No data available for plotting: {split}")
            return

        pred_array = np.array(self.predictions[split])
        target_array = np.array(self.targets[split])
        errors = pred_array - target_array

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Error Analysis - {split.capitalize()} Set', fontsize=16)

        # 1. Predicted vs Actual
        axes[0, 0].scatter(target_array, pred_array, alpha=0.6, s=10)
        axes[0, 0].plot([target_array.min(), target_array.max()], [target_array.min(), target_array.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual HOMO-LUMO Gap (eV)')
        axes[0, 0].set_ylabel('Predicted HOMO-LUMO Gap (eV)')
        axes[0, 0].set_title('Predicted vs Actual')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Error distribution
        axes[0, 1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Error (eV)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Residual plot
        axes[1, 0].scatter(target_array, errors, alpha=0.6, s=10)
        axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Actual HOMO-LUMO Gap (eV)')
        axes[1, 0].set_ylabel('Residual (eV)')
        axes[1, 0].set_title('Residuals vs Actual')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Absolute error vs target value
        abs_errors = np.abs(errors)
        axes[1, 1].scatter(target_array, abs_errors, alpha=0.6, s=10)
        axes[1, 1].axhline(self.target_mae_ev, color='red', linestyle='--', linewidth=2,
                          label=f'Target MAE: {self.target_mae_ev:.3f} eV')
        axes[1, 1].set_xlabel('Actual HOMO-LUMO Gap (eV)')
        axes[1, 1].set_ylabel('Absolute Error (eV)')
        axes[1, 1].set_title('Absolute Error vs Actual')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Error plots saved to {save_path}")

        plt.close()


class ConvergenceAnalyzer:
    """Analyzer for training convergence and curriculum learning effectiveness."""

    def __init__(self, baseline_reference: Optional[float] = None):
        """Initialize convergence analyzer.

        Args:
            baseline_reference: Reference baseline performance for speedup calculation
        """
        self.baseline_reference = baseline_reference
        self.training_history = []

    def add_epoch_data(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_mae: float,
        val_mae: float,
        curriculum_fraction: float,
        lr: float
    ) -> None:
        """Add training data for one epoch.

        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            train_mae: Training MAE
            val_mae: Validation MAE
            curriculum_fraction: Fraction of data used in curriculum
            lr: Learning rate
        """
        self.training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'curriculum_fraction': curriculum_fraction,
            'lr': lr
        })

    def compute_convergence_metrics(self) -> Dict[str, float]:
        """Compute convergence speed and stability metrics.

        Returns:
            Dictionary with convergence metrics
        """
        if len(self.training_history) < 10:
            logger.warning("Insufficient training history for convergence analysis")
            return {}

        val_maes = [entry['val_mae'] for entry in self.training_history]
        epochs = [entry['epoch'] for entry in self.training_history]

        metrics = {}

        # Find convergence point (epoch where val_mae reaches 95% of final performance)
        final_mae = val_maes[-5:]  # Use last 5 epochs for final performance
        target_mae = np.mean(final_mae) / 0.95

        convergence_epoch = None
        for i, mae in enumerate(val_maes):
            if mae <= target_mae:
                convergence_epoch = epochs[i]
                break

        if convergence_epoch is not None:
            metrics['convergence_epoch'] = float(convergence_epoch)

            # Convergence speedup vs baseline
            if self.baseline_reference is not None:
                speedup = self.baseline_reference / convergence_epoch
                metrics['convergence_speedup_vs_baseline'] = float(speedup)

        # Training stability (variance in last 20% of training)
        last_portion = max(1, len(val_maes) // 5)
        recent_maes = val_maes[-last_portion:]
        metrics['training_stability'] = float(np.std(recent_maes))

        # Best validation MAE achieved
        metrics['best_val_mae'] = float(np.min(val_maes))
        metrics['final_val_mae'] = float(np.mean(val_maes[-3:]))  # Last 3 epochs

        # Overfitting detection
        train_maes = [entry['train_mae'] for entry in self.training_history]
        final_train_mae = np.mean(train_maes[-3:])
        final_val_mae = np.mean(val_maes[-3:])
        metrics['generalization_gap'] = float(final_val_mae - final_train_mae)

        return metrics

    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """Plot training curves with curriculum information.

        Args:
            save_path: Optional path to save plot
        """
        if len(self.training_history) < 2:
            logger.warning("Insufficient data for plotting")
            return

        epochs = [entry['epoch'] for entry in self.training_history]
        train_losses = [entry['train_loss'] for entry in self.training_history]
        val_losses = [entry['val_loss'] for entry in self.training_history]
        train_maes = [entry['train_mae'] for entry in self.training_history]
        val_maes = [entry['val_mae'] for entry in self.training_history]
        curriculum_fractions = [entry['curriculum_fraction'] for entry in self.training_history]
        lrs = [entry['lr'] for entry in self.training_history]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress with Curriculum Learning', fontsize=16)

        # 1. Loss curves
        axes[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, val_losses, label='Val Loss', color='orange')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. MAE curves
        axes[0, 1].plot(epochs, train_maes, label='Train MAE', color='blue')
        axes[0, 1].plot(epochs, val_maes, label='Val MAE', color='orange')
        if self.baseline_reference is not None:
            axes[0, 1].axhline(self.baseline_reference, color='red', linestyle='--',
                              label=f'Baseline: {self.baseline_reference:.3f}')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE (eV)')
        axes[0, 1].set_title('Training and Validation MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Curriculum progression
        axes[1, 0].plot(epochs, curriculum_fractions, color='green', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Data Fraction Used')
        axes[1, 0].set_title('Curriculum Learning Progress')
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Learning rate schedule
        axes[1, 1].plot(epochs, lrs, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")

        plt.close()


class ErrorAnalyzer:
    """Analyzer for detailed error patterns and failure modes."""

    def __init__(self):
        """Initialize error analyzer."""
        self.error_data = []

    def analyze_prediction_errors(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        molecular_features: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Analyze prediction errors in detail.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            molecular_features: Optional molecular features for correlation analysis

        Returns:
            Detailed error analysis results
        """
        errors = predictions - targets
        abs_errors = np.abs(errors)

        analysis = {
            'summary': {
                'mae': float(np.mean(abs_errors)),
                'rmse': float(np.sqrt(np.mean(errors**2))),
                'bias': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'max_error': float(np.max(abs_errors)),
                'min_error': float(np.min(abs_errors))
            },
            'distribution': {
                'skewness': float(stats.skew(errors)),
                'kurtosis': float(stats.kurtosis(errors)),
                'percentiles': {
                    'p10': float(np.percentile(abs_errors, 10)),
                    'p25': float(np.percentile(abs_errors, 25)),
                    'p50': float(np.percentile(abs_errors, 50)),
                    'p75': float(np.percentile(abs_errors, 75)),
                    'p90': float(np.percentile(abs_errors, 90)),
                    'p95': float(np.percentile(abs_errors, 95)),
                    'p99': float(np.percentile(abs_errors, 99))
                }
            }
        }

        # Analyze errors by target value ranges
        target_ranges = [
            (0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, float('inf'))
        ]

        range_analysis = {}
        for low, high in target_ranges:
            mask = (targets >= low) & (targets < high)
            if np.any(mask):
                range_key = f"{low}-{high if high != float('inf') else 'inf'}_eV"
                range_analysis[range_key] = {
                    'count': int(np.sum(mask)),
                    'mae': float(np.mean(abs_errors[mask])),
                    'bias': float(np.mean(errors[mask])),
                    'std': float(np.std(errors[mask]))
                }

        analysis['range_analysis'] = range_analysis

        # Feature correlation analysis (if features provided)
        if molecular_features is not None:
            feature_correlations = {}
            for feature_name, feature_values in molecular_features.items():
                if len(feature_values) == len(abs_errors):
                    try:
                        corr, p_value = stats.pearsonr(feature_values, abs_errors)
                        feature_correlations[feature_name] = {
                            'correlation': float(corr),
                            'p_value': float(p_value),
                            'significant': bool(p_value < 0.05)
                        }
                    except Exception as e:
                        logger.warning(f"Failed to compute correlation for {feature_name}: {e}")

            analysis['feature_correlations'] = feature_correlations

        return analysis


class StatisticalSignificanceTester:
    """Statistical significance testing for model comparisons."""

    @staticmethod
    def compare_models(
        results_a: Dict[str, float],
        results_b: Dict[str, float],
        metric: str = 'mae',
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Compare two models using statistical tests.

        Args:
            results_a: Results from model A
            results_b: Results from model B
            metric: Metric to compare
            alpha: Significance level

        Returns:
            Statistical comparison results
        """
        # This is a simplified version - in practice, you'd need raw predictions
        # for proper statistical testing

        mae_a = results_a.get(metric, float('inf'))
        mae_b = results_b.get(metric, float('inf'))

        improvement = (mae_a - mae_b) / mae_a if mae_a > 0 else 0.0

        return {
            'metric': metric,
            'model_a_score': mae_a,
            'model_b_score': mae_b,
            'improvement': improvement,
            'significant': abs(improvement) > 0.05,  # Heuristic threshold
            'better_model': 'B' if mae_b < mae_a else 'A'
        }

    @staticmethod
    def curriculum_effectiveness_test(
        curriculum_results: List[float],
        baseline_results: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Test effectiveness of curriculum learning.

        Args:
            curriculum_results: Results with curriculum learning
            baseline_results: Results without curriculum learning
            alpha: Significance level

        Returns:
            Statistical test results
        """
        try:
            # Paired t-test (if same number of runs)
            if len(curriculum_results) == len(baseline_results):
                statistic, p_value = stats.ttest_rel(baseline_results, curriculum_results)
                test_name = "Paired t-test"
            else:
                # Independent t-test
                statistic, p_value = stats.ttest_ind(baseline_results, curriculum_results)
                test_name = "Independent t-test"

            significant = p_value < alpha

            improvement = (np.mean(baseline_results) - np.mean(curriculum_results)) / np.mean(baseline_results)

            return {
                'test_name': test_name,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': significant,
                'alpha': alpha,
                'improvement': float(improvement),
                'curriculum_mean': float(np.mean(curriculum_results)),
                'baseline_mean': float(np.mean(baseline_results)),
                'curriculum_std': float(np.std(curriculum_results)),
                'baseline_std': float(np.std(baseline_results))
            }

        except Exception as e:
            logger.error(f"Statistical test failed: {e}")
            return {'error': str(e)}