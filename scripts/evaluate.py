#!/usr/bin/env python3
"""Evaluation script for trained Spectral-Temporal models.

This script performs comprehensive evaluation including:
- Test set performance
- Error analysis and breakdown
- Statistical significance testing
- Model interpretability analysis
- Performance visualization
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning import Trainer
import mlflow
import mlflow.pytorch

from spectral_temporal_curriculum_molecular_gap_prediction.utils.config import load_config, Config
from spectral_temporal_curriculum_molecular_gap_prediction.data.loader import PCQM4Mv2DataModule
from spectral_temporal_curriculum_molecular_gap_prediction.training.trainer import CurriculumTrainer
from spectral_temporal_curriculum_molecular_gap_prediction.evaluation.metrics import (
    MolecularGapMetrics, ErrorAnalyzer, StatisticalSignificanceTester
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluator with analysis and visualization."""

    def __init__(self, config: Config):
        """Initialize evaluator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.metrics = MolecularGapMetrics(
            target_mae_ev=config.evaluation.target_mae_ev,
            tail_percentile=config.evaluation.tail_percentile,
            compute_correlations=config.evaluation.compute_correlations
        )
        self.error_analyzer = ErrorAnalyzer()

    def load_model_from_checkpoint(self, checkpoint_path: str) -> CurriculumTrainer:
        """Load trained model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint

        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {checkpoint_path}")

        # Load model
        model = CurriculumTrainer.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_config=vars(self.config.model),
            optimizer_config=vars(self.config.optimizer),
            scheduler_config=vars(self.config.scheduler),
            curriculum_config=vars(self.config.curriculum),
            loss_config=vars(self.config.loss),
            evaluation_config=vars(self.config.evaluation)
        )

        model.eval()
        logger.info("Model loaded successfully")
        return model

    def evaluate_test_set(
        self,
        model: CurriculumTrainer,
        datamodule: PCQM4Mv2DataModule
    ) -> Dict[str, Any]:
        """Evaluate model on test set.

        Args:
            model: Trained model
            datamodule: Data module

        Returns:
            Test results dictionary
        """
        logger.info("Evaluating on test set...")

        # Setup test data
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()

        # Create trainer for testing
        trainer = Trainer(
            accelerator=self.config.device,
            devices=1,
            logger=False,
            enable_progress_bar=True
        )

        # Run test
        test_results = trainer.test(model, test_loader, verbose=True)

        # Collect all predictions for detailed analysis
        model.eval()
        all_predictions = []
        all_targets = []
        all_attention_weights = []

        with torch.no_grad():
            for batch in test_loader:
                # Forward pass
                predictions = model(batch)
                targets = batch.y.view(-1, 1)

                # Get attention weights for interpretability
                attention_weights = model.model.get_attention_weights(batch)

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_attention_weights.append(attention_weights)

        # Concatenate results
        predictions_array = np.concatenate(all_predictions, axis=0)
        targets_array = np.concatenate(all_targets, axis=0)

        # Compute detailed metrics
        detailed_metrics = self._compute_detailed_metrics(predictions_array, targets_array)

        results = {
            'test_metrics': test_results[0] if test_results else {},
            'detailed_metrics': detailed_metrics,
            'predictions': predictions_array,
            'targets': targets_array,
            'attention_weights': all_attention_weights,
            'num_samples': len(predictions_array)
        }

        return results

    def _compute_detailed_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, Any]:
        """Compute detailed evaluation metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Detailed metrics dictionary
        """
        # Reset and update metrics
        self.metrics.reset('test')

        # Convert to torch tensors for metric computation
        pred_tensor = torch.from_numpy(predictions)
        target_tensor = torch.from_numpy(targets)

        self.metrics.update(pred_tensor, target_tensor, split='test')
        standard_metrics = self.metrics.compute(split='test')

        # Error analysis
        error_analysis = self.metrics.get_error_analysis(split='test')

        # Additional molecular-specific analysis
        molecular_analysis = self._analyze_molecular_performance(predictions, targets)

        return {
            'standard_metrics': standard_metrics,
            'error_analysis': error_analysis,
            'molecular_analysis': molecular_analysis
        }

    def _analyze_molecular_performance(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze performance on different types of molecules.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Analysis results by molecular characteristics
        """
        errors = predictions.flatten() - targets.flatten()
        abs_errors = np.abs(errors)

        # Analyze by HOMO-LUMO gap ranges (typical for different molecule types)
        gap_ranges = [
            (0.0, 2.0, "Small gap (metallic)"),
            (2.0, 4.0, "Medium gap (semiconductor)"),
            (4.0, 6.0, "Large gap (insulator)"),
            (6.0, 8.0, "Very large gap"),
            (8.0, float('inf'), "Extreme gap")
        ]

        range_analysis = {}
        for low, high, description in gap_ranges:
            mask = (targets.flatten() >= low) & (targets.flatten() < high)
            if np.any(mask):
                range_errors = abs_errors[mask]
                range_analysis[f"{low:.1f}-{high:.1f}_eV"] = {
                    'description': description,
                    'count': int(np.sum(mask)),
                    'fraction': float(np.mean(mask)),
                    'mae': float(np.mean(range_errors)),
                    'mae_std': float(np.std(range_errors)),
                    'max_error': float(np.max(range_errors)),
                    'p95_error': float(np.percentile(range_errors, 95))
                }

        # Performance vs target quality assessment
        target_mae = self.config.evaluation.target_mae_ev
        performance_assessment = {
            'target_mae_ev': target_mae,
            'achieved_mae_ev': float(np.mean(abs_errors)),
            'target_achieved': bool(np.mean(abs_errors) <= target_mae),
            'samples_within_target': int(np.sum(abs_errors <= target_mae)),
            'fraction_within_target': float(np.mean(abs_errors <= target_mae))
        }

        return {
            'gap_range_analysis': range_analysis,
            'performance_assessment': performance_assessment
        }

    def create_evaluation_report(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """Create comprehensive evaluation report.

        Args:
            results: Evaluation results
            save_path: Optional path to save report

        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("SPECTRAL-TEMPORAL CURRICULUM LEARNING MODEL EVALUATION REPORT")
        report_lines.append("="*80)

        # Model and data information
        report_lines.append("\nMODEL INFORMATION:")
        report_lines.append("-" * 50)
        report_lines.append(f"Architecture: Dual-view Spectral-Temporal Network")
        report_lines.append(f"Test samples: {results['num_samples']:,}")
        report_lines.append(f"Target MAE: {self.config.evaluation.target_mae_ev:.3f} eV")

        # Main performance metrics
        metrics = results['detailed_metrics']['standard_metrics']
        report_lines.append("\nPERFORMANCE METRICS:")
        report_lines.append("-" * 50)
        report_lines.append(f"Test MAE: {metrics.get('mae', 0):.4f} eV")
        report_lines.append(f"Test RMSE: {metrics.get('rmse', 0):.4f} eV")

        if 'r2' in metrics:
            report_lines.append(f"R²: {metrics['r2']:.4f}")
        if 'pearson_r' in metrics:
            report_lines.append(f"Pearson correlation: {metrics['pearson_r']:.4f}")

        # Target achievement
        performance = results['detailed_metrics']['molecular_analysis']['performance_assessment']
        if performance['target_achieved']:
            report_lines.append(f"✓ TARGET ACHIEVED: {performance['achieved_mae_ev']:.4f} ≤ {performance['target_mae_ev']:.4f} eV")
        else:
            report_lines.append(f"✗ Target not achieved: {performance['achieved_mae_ev']:.4f} > {performance['target_mae_ev']:.4f} eV")

        # Error distribution
        report_lines.append("\nERROR DISTRIBUTION:")
        report_lines.append("-" * 50)
        report_lines.append(f"50th percentile: {metrics.get('mae_p50', 0):.4f} eV")
        report_lines.append(f"90th percentile: {metrics.get('mae_p90', 0):.4f} eV")
        report_lines.append(f"95th percentile: {metrics.get('mae_p95', 0):.4f} eV")
        report_lines.append(f"99th percentile: {metrics.get('mae_p99', 0):.4f} eV")
        report_lines.append(f"Maximum error: {metrics.get('mae_max', 0):.4f} eV")

        # Molecular type analysis
        gap_analysis = results['detailed_metrics']['molecular_analysis']['gap_range_analysis']
        if gap_analysis:
            report_lines.append("\nPERFORMANCE BY MOLECULE TYPE:")
            report_lines.append("-" * 50)
            for gap_range, stats in gap_analysis.items():
                report_lines.append(f"{gap_range} ({stats['description']}):")
                report_lines.append(f"  Samples: {stats['count']:,} ({stats['fraction']:.1%})")
                report_lines.append(f"  MAE: {stats['mae']:.4f} ± {stats['mae_std']:.4f} eV")
                report_lines.append(f"  95th percentile: {stats['p95_error']:.4f} eV")

        # Error analysis
        error_analysis = results['detailed_metrics']['error_analysis']
        if 'outliers' in error_analysis:
            outliers = error_analysis['outliers']
            report_lines.append("\nOUTLIER ANALYSIS:")
            report_lines.append("-" * 50)
            report_lines.append(f"Outlier threshold: {outliers['threshold']:.4f} eV")
            report_lines.append(f"Outlier count: {outliers['count']:,} ({outliers['fraction']:.1%})")
            if outliers['count'] > 0:
                report_lines.append(f"Mean outlier error: {outliers['mean_error']:.4f} eV")

        # Research target metrics
        report_lines.append("\nRESEARCH TARGET METRICS:")
        report_lines.append("-" * 50)
        report_lines.append(f"MAE (target ≤ 0.082 eV): {metrics.get('mae', 0):.4f} eV")

        tail_mae = metrics.get('mae_p95', 0)
        report_lines.append(f"Tail 95th percentile MAE (target ≤ 0.14 eV): {tail_mae:.4f} eV")

        # Add convergence info if available
        if 'convergence_speedup_vs_baseline' in metrics:
            speedup = metrics['convergence_speedup_vs_baseline']
            report_lines.append(f"Convergence speedup vs baseline (target ≥ 1.35x): {speedup:.2f}x")

        report_lines.append("\n" + "="*80)

        report_text = "\n".join(report_lines)

        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {save_path}")

        return report_text

    def create_visualizations(
        self,
        results: Dict[str, Any],
        save_dir: str = "evaluation_plots"
    ) -> List[str]:
        """Create evaluation visualizations.

        Args:
            results: Evaluation results
            save_dir: Directory to save plots

        Returns:
            List of saved plot paths
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        plot_paths = []
        predictions = results['predictions'].flatten()
        targets = results['targets'].flatten()

        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Prediction vs Actual scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(targets, predictions, alpha=0.6, s=20)
        min_val, max_val = min(targets.min(), predictions.min()), max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

        plt.xlabel('Actual HOMO-LUMO Gap (eV)', fontsize=12)
        plt.ylabel('Predicted HOMO-LUMO Gap (eV)', fontsize=12)
        plt.title('Predicted vs Actual HOMO-LUMO Gap', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add R² and MAE to plot
        r2 = np.corrcoef(targets, predictions)[0, 1] ** 2
        mae = np.mean(np.abs(targets - predictions))
        plt.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.3f} eV',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plot_path = save_dir / "prediction_vs_actual.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))

        # 2. Error distribution histogram
        errors = predictions - targets
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Prediction Error (eV)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        abs_errors = np.abs(errors)
        plt.hist(abs_errors, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(self.config.evaluation.target_mae_ev, color='red',
                   linestyle='--', linewidth=2, label=f'Target MAE: {self.config.evaluation.target_mae_ev:.3f} eV')
        plt.xlabel('Absolute Error (eV)')
        plt.ylabel('Frequency')
        plt.title('Absolute Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = save_dir / "error_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))

        # 3. Performance by molecule type
        gap_analysis = results['detailed_metrics']['molecular_analysis']['gap_range_analysis']
        if gap_analysis:
            gap_ranges = list(gap_analysis.keys())
            maes = [gap_analysis[gr]['mae'] for gr in gap_ranges]
            counts = [gap_analysis[gr]['count'] for gr in gap_ranges]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # MAE by gap range
            bars = ax1.bar(range(len(gap_ranges)), maes, alpha=0.7)
            ax1.set_xlabel('HOMO-LUMO Gap Range (eV)')
            ax1.set_ylabel('MAE (eV)')
            ax1.set_title('Performance by Molecular Type')
            ax1.set_xticks(range(len(gap_ranges)))
            ax1.set_xticklabels([gr.replace('_eV', '') for gr in gap_ranges], rotation=45)
            ax1.axhline(self.config.evaluation.target_mae_ev, color='red',
                       linestyle='--', alpha=0.7, label=f'Target: {self.config.evaluation.target_mae_ev:.3f} eV')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Sample counts
            ax2.bar(range(len(gap_ranges)), counts, alpha=0.7, color='orange')
            ax2.set_xlabel('HOMO-LUMO Gap Range (eV)')
            ax2.set_ylabel('Number of Samples')
            ax2.set_title('Sample Distribution by Molecular Type')
            ax2.set_xticks(range(len(gap_ranges)))
            ax2.set_xticklabels([gr.replace('_eV', '') for gr in gap_ranges], rotation=45)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = save_dir / "performance_by_molecule_type.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(str(plot_path))

        # 4. Error percentile plot
        percentiles = np.arange(1, 101)
        abs_errors = np.abs(predictions - targets)
        error_percentiles = [np.percentile(abs_errors, p) for p in percentiles]

        plt.figure(figsize=(10, 6))
        plt.plot(percentiles, error_percentiles, 'b-', linewidth=2)
        plt.axhline(self.config.evaluation.target_mae_ev, color='red',
                   linestyle='--', linewidth=2, label=f'Target MAE: {self.config.evaluation.target_mae_ev:.3f} eV')
        plt.axvline(95, color='green', linestyle='--', alpha=0.7, label='95th percentile')

        plt.xlabel('Percentile')
        plt.ylabel('Absolute Error (eV)')
        plt.title('Error Percentile Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = save_dir / "error_percentiles.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))

        logger.info(f"Created {len(plot_paths)} evaluation plots in {save_dir}")
        return plot_paths


def compare_with_baseline(
    test_results: Dict[str, Any],
    baseline_mae: float = 0.11
) -> Dict[str, Any]:
    """Compare results with baseline performance.

    Args:
        test_results: Test evaluation results
        baseline_mae: Baseline MAE for comparison

    Returns:
        Comparison results
    """
    achieved_mae = test_results['detailed_metrics']['standard_metrics']['mae']

    comparison = {
        'baseline_mae': baseline_mae,
        'achieved_mae': achieved_mae,
        'improvement': (baseline_mae - achieved_mae) / baseline_mae,
        'improvement_absolute': baseline_mae - achieved_mae,
        'better_than_baseline': achieved_mae < baseline_mae
    }

    return comparison


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Spectral-Temporal Curriculum Learning Model")

    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--baseline_mae', type=float, default=0.11,
                      help='Baseline MAE for comparison')

    # MLflow arguments
    parser.add_argument('--mlflow_run_id', type=str, default=None,
                      help='MLflow run ID to log results to')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load configuration
    config = load_config(args.config)

    # Initialize evaluator
    evaluator = ModelEvaluator(config)

    try:
        # Load model
        model = evaluator.load_model_from_checkpoint(args.checkpoint)

        # Initialize data module
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

        # Evaluate on test set
        logger.info("Running comprehensive evaluation...")
        test_results = evaluator.evaluate_test_set(model, datamodule)

        # Compare with baseline
        baseline_comparison = compare_with_baseline(test_results, args.baseline_mae)

        # Create evaluation report
        report_path = output_dir / "evaluation_report.txt"
        report_text = evaluator.create_evaluation_report(test_results, str(report_path))
        print(report_text)

        # Create visualizations
        plot_paths = evaluator.create_visualizations(test_results, str(output_dir / "plots"))

        # Save detailed results
        results_summary = {
            'test_metrics': test_results['detailed_metrics']['standard_metrics'],
            'baseline_comparison': baseline_comparison,
            'num_test_samples': test_results['num_samples']
        }

        # Log to MLflow if run ID provided
        if args.mlflow_run_id:
            with mlflow.start_run(run_id=args.mlflow_run_id):
                # Log test metrics
                for key, value in results_summary['test_metrics'].items():
                    mlflow.log_metric(f"test_{key}", value)

                # Log baseline comparison
                mlflow.log_metric("baseline_mae", baseline_comparison['baseline_mae'])
                mlflow.log_metric("improvement_vs_baseline", baseline_comparison['improvement'])

                # Log evaluation artifacts
                mlflow.log_artifact(str(report_path))
                for plot_path in plot_paths:
                    mlflow.log_artifact(plot_path)

        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Test MAE: {results_summary['test_metrics']['mae']:.4f} eV")
        print(f"Baseline MAE: {baseline_comparison['baseline_mae']:.4f} eV")
        print(f"Improvement: {baseline_comparison['improvement']:.1%}")

        target_mae = config.evaluation.target_mae_ev
        achieved_mae = results_summary['test_metrics']['mae']

        if achieved_mae <= target_mae:
            print(f"✓ Research target achieved: {achieved_mae:.4f} ≤ {target_mae:.4f} eV")
        else:
            print(f"✗ Research target not achieved: {achieved_mae:.4f} > {target_mae:.4f} eV")

        print(f"\nResults saved to: {output_dir}")
        print("="*60)

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())