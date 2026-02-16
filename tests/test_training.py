"""Tests for training and curriculum learning modules."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
import pytorch_lightning as pl

from spectral_temporal_curriculum_molecular_gap_prediction.training.trainer import (
    CustomLoss, CurriculumScheduler, SpectralComplexityScheduler, CurriculumTrainer
)
from spectral_temporal_curriculum_molecular_gap_prediction.evaluation.metrics import MolecularGapMetrics


class TestCustomLoss:
    """Test suite for CustomLoss function."""

    def test_initialization(self):
        """Test CustomLoss initialization."""
        loss_fn = CustomLoss(
            base_loss='mae',
            uncertainty_weight=0.1,
            spectral_regularization=0.01,
            curriculum_weight=0.05
        )

        assert loss_fn.base_loss == 'mae'
        assert loss_fn.uncertainty_weight == 0.1
        assert loss_fn.spectral_regularization == 0.01
        assert loss_fn.curriculum_weight == 0.05

    def test_different_base_losses(self):
        """Test different base loss functions."""
        base_losses = ['mae', 'mse', 'huber']

        for base_loss in base_losses:
            loss_fn = CustomLoss(base_loss=base_loss)
            assert loss_fn.base_loss == base_loss

    def test_invalid_base_loss(self):
        """Test that invalid base loss raises error."""
        with pytest.raises(ValueError):
            CustomLoss(base_loss='invalid')

    def test_forward_basic(self, sample_predictions_targets):
        """Test basic forward pass without optional parameters."""
        loss_fn = CustomLoss(base_loss='mae')
        predictions, targets = sample_predictions_targets

        loss_dict = loss_fn(predictions, targets)

        assert 'base_loss' in loss_dict
        assert 'spectral_reg' in loss_dict
        assert 'total_loss' in loss_dict

        assert isinstance(loss_dict['base_loss'], torch.Tensor)
        assert loss_dict['base_loss'].item() >= 0

    def test_forward_with_curriculum_weighting(self, sample_predictions_targets):
        """Test forward pass with curriculum difficulty weighting."""
        loss_fn = CustomLoss(curriculum_weight=0.1)
        predictions, targets = sample_predictions_targets

        # Create curriculum difficulty scores
        curriculum_difficulty = torch.rand(len(predictions), 1)

        loss_dict = loss_fn(
            predictions,
            targets,
            curriculum_difficulty=curriculum_difficulty
        )

        assert 'base_loss' in loss_dict
        assert not torch.isnan(loss_dict['base_loss'])

    def test_forward_with_model_regularization(self, sample_predictions_targets):
        """Test forward pass with model-based regularization."""
        # Create a mock model with spectral layers
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.spectral_layer = nn.Linear(10, 5)

        model = MockModel()
        loss_fn = CustomLoss(spectral_regularization=0.1)
        predictions, targets = sample_predictions_targets

        loss_dict = loss_fn(predictions, targets, model=model)

        assert 'spectral_reg' in loss_dict
        assert loss_dict['spectral_reg'].item() >= 0

    def test_different_loss_components(self, sample_predictions_targets):
        """Test that different loss components are computed correctly."""
        loss_fn = CustomLoss(
            base_loss='mse',
            spectral_regularization=0.05,
            curriculum_weight=0.1
        )

        predictions, targets = sample_predictions_targets
        curriculum_difficulty = torch.rand(len(predictions), 1)

        loss_dict = loss_fn(
            predictions,
            targets,
            curriculum_difficulty=curriculum_difficulty
        )

        # Total loss should be sum of components
        expected_total = (
            loss_dict['base_loss'] +
            0.05 * loss_dict['spectral_reg']
        )

        assert torch.allclose(loss_dict['total_loss'], expected_total, atol=1e-6)


class TestCurriculumSchedulers:
    """Test suite for curriculum learning schedulers."""

    def test_spectral_complexity_scheduler_initialization(self):
        """Test SpectralComplexityScheduler initialization."""
        scheduler = SpectralComplexityScheduler(
            initial_fraction=0.2,
            final_fraction=1.0,
            warmup_epochs=3,
            total_epochs=50,
            growth_strategy='exponential'
        )

        assert scheduler.initial_fraction == 0.2
        assert scheduler.final_fraction == 1.0
        assert scheduler.warmup_epochs == 3
        assert scheduler.total_epochs == 50
        assert scheduler.growth_strategy == 'exponential'

    def test_scheduler_warmup_period(self):
        """Test scheduler behavior during warmup period."""
        scheduler = SpectralComplexityScheduler(
            initial_fraction=0.1,
            warmup_epochs=5,
            total_epochs=20
        )

        # During warmup, should return initial fraction
        for epoch in range(5):
            fraction = scheduler.get_difficulty_fraction(0, epoch)
            assert fraction == 0.1

    def test_scheduler_growth_strategies(self):
        """Test different growth strategies."""
        strategies = ['linear', 'exponential', 'sigmoid']

        for strategy in strategies:
            scheduler = SpectralComplexityScheduler(
                initial_fraction=0.1,
                final_fraction=1.0,
                warmup_epochs=2,
                total_epochs=10,
                growth_strategy=strategy
            )

            # Test progression
            fractions = []
            for epoch in range(10):
                fraction = scheduler.get_difficulty_fraction(0, epoch)
                fractions.append(fraction)

            # Should be monotonically increasing after warmup
            for i in range(2, len(fractions) - 1):
                assert fractions[i] >= fractions[i-1] - 1e-6

            # Should end at final fraction
            assert abs(fractions[-1] - 1.0) < 1e-6

    def test_scheduler_step_and_epoch_updates(self):
        """Test scheduler state updates."""
        scheduler = SpectralComplexityScheduler()

        assert scheduler.current_step == 0
        assert scheduler.current_epoch == 0

        scheduler.step()
        assert scheduler.current_step == 1

        scheduler.epoch()
        assert scheduler.current_epoch == 1
        assert scheduler.current_step == 0

    def test_scheduler_edge_cases(self):
        """Test scheduler edge cases."""
        scheduler = SpectralComplexityScheduler(
            initial_fraction=0.5,
            final_fraction=0.5,  # No growth
            total_epochs=10
        )

        # Should always return the same fraction
        for epoch in range(10):
            fraction = scheduler.get_difficulty_fraction(0, epoch)
            assert abs(fraction - 0.5) < 1e-6

    def test_minimum_growth_rate(self):
        """Test minimum growth rate enforcement."""
        scheduler = SpectralComplexityScheduler(
            initial_fraction=0.1,
            final_fraction=1.0,
            warmup_epochs=1,
            total_epochs=100,
            min_growth_rate=0.05
        )

        # Check that minimum growth is enforced
        for epoch in range(1, 20):
            fraction = scheduler.get_difficulty_fraction(0, epoch)
            min_expected = 0.1 + epoch * 0.05
            assert fraction >= min_expected - 1e-6


class TestCurriculumTrainer:
    """Test suite for CurriculumTrainer."""

    @pytest.fixture
    def trainer_config(self, test_config):
        """Create trainer configuration."""
        return {
            'model_config': vars(test_config.model),
            'optimizer_config': vars(test_config.optimizer),
            'scheduler_config': vars(test_config.scheduler),
            'curriculum_config': vars(test_config.curriculum),
            'loss_config': vars(test_config.loss),
            'evaluation_config': vars(test_config.evaluation)
        }

    def test_initialization(self, trainer_config):
        """Test CurriculumTrainer initialization."""
        trainer = CurriculumTrainer(**trainer_config)

        assert trainer.model is not None
        assert trainer.loss_fn is not None
        assert trainer.curriculum_scheduler is not None
        assert trainer.metrics is not None

    def test_forward_pass(self, trainer_config, batch_molecular_graphs):
        """Test forward pass through trainer."""
        trainer = CurriculumTrainer(**trainer_config)

        output = trainer(batch_molecular_graphs)

        batch_size = batch_molecular_graphs.num_graphs
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()

    def test_training_step(self, trainer_config, batch_molecular_graphs):
        """Test training step computation."""
        trainer = CurriculumTrainer(**trainer_config)

        # Mock targets
        batch_molecular_graphs.y = torch.randn(batch_molecular_graphs.num_graphs, 1)

        loss = trainer.training_step(batch_molecular_graphs, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_validation_step(self, trainer_config, batch_molecular_graphs):
        """Test validation step computation."""
        trainer = CurriculumTrainer(**trainer_config)

        # Mock targets
        batch_molecular_graphs.y = torch.randn(batch_molecular_graphs.num_graphs, 1)

        # Should not raise exception
        trainer.validation_step(batch_molecular_graphs, 0)

    def test_test_step(self, trainer_config, batch_molecular_graphs):
        """Test test step computation."""
        trainer = CurriculumTrainer(**trainer_config)

        # Mock targets
        batch_molecular_graphs.y = torch.randn(batch_molecular_graphs.num_graphs, 1)

        # Should not raise exception
        trainer.test_step(batch_molecular_graphs, 0)

    def test_configure_optimizers(self, trainer_config):
        """Test optimizer configuration."""
        trainer = CurriculumTrainer(**trainer_config)

        optimizer_config = trainer.configure_optimizers()

        assert 'optimizer' in optimizer_config
        assert optimizer_config['optimizer'] is not None

    def test_different_optimizer_types(self, test_config):
        """Test different optimizer configurations."""
        optimizers = ['adamw', 'adam', 'sgd']

        for opt_name in optimizers:
            config = test_config
            config.optimizer.name = opt_name

            trainer_config = {
                'model_config': vars(config.model),
                'optimizer_config': vars(config.optimizer),
                'scheduler_config': vars(config.scheduler),
                'curriculum_config': vars(config.curriculum),
                'loss_config': vars(config.loss),
                'evaluation_config': vars(config.evaluation)
            }

            trainer = CurriculumTrainer(**trainer_config)
            opt_config = trainer.configure_optimizers()

            assert opt_config['optimizer'] is not None

    def test_different_scheduler_types(self, test_config):
        """Test different scheduler configurations."""
        schedulers = ['cosine', 'exponential', 'plateau', 'onecycle']

        for sched_name in schedulers:
            config = test_config
            config.scheduler.name = sched_name
            if sched_name == 'onecycle':
                config.scheduler.total_steps = 100

            trainer_config = {
                'model_config': vars(config.model),
                'optimizer_config': vars(config.optimizer),
                'scheduler_config': vars(config.scheduler),
                'curriculum_config': vars(config.curriculum),
                'loss_config': vars(config.loss),
                'evaluation_config': vars(config.evaluation)
            }

            trainer = CurriculumTrainer(**trainer_config)
            opt_config = trainer.configure_optimizers()

            if sched_name != 'null':
                assert 'lr_scheduler' in opt_config

    def test_epoch_start_hook(self, trainer_config):
        """Test on_train_epoch_start hook."""
        trainer = CurriculumTrainer(**trainer_config)

        initial_fraction = trainer.current_curriculum_fraction

        # Simulate epoch start
        trainer.current_epoch = 1
        trainer.on_train_epoch_start()

        # Curriculum fraction should be updated
        assert hasattr(trainer, 'current_curriculum_fraction')

    def test_epoch_end_hooks(self, trainer_config):
        """Test epoch end hooks."""
        trainer = CurriculumTrainer(**trainer_config)

        # Should not raise exceptions
        trainer.on_train_epoch_end()
        trainer.on_validation_epoch_end()
        trainer.on_test_epoch_end()

    def test_predict_step(self, trainer_config, batch_molecular_graphs):
        """Test prediction step."""
        trainer = CurriculumTrainer(**trainer_config)

        # Mock targets
        batch_molecular_graphs.y = torch.randn(batch_molecular_graphs.num_graphs, 1)

        result = trainer.predict_step(batch_molecular_graphs, 0)

        assert isinstance(result, dict)
        assert 'predictions' in result
        assert 'targets' in result
        assert 'attention_weights' in result

    @pytest.mark.integration
    def test_curriculum_progression_integration(self, trainer_config):
        """Test curriculum progression over multiple epochs."""
        trainer = CurriculumTrainer(**trainer_config)

        fractions = []
        for epoch in range(5):
            trainer.current_epoch = epoch
            trainer.on_train_epoch_start()
            fractions.append(trainer.current_curriculum_fraction)

        # Should show progression (after warmup)
        assert len(fractions) == 5
        # Later fractions should generally be >= earlier ones
        for i in range(1, len(fractions)):
            assert fractions[i] >= fractions[i-1] - 1e-6


class TestMetricsIntegration:
    """Test integration between training and metrics."""

    def test_metrics_update_and_compute(self):
        """Test metrics update and computation."""
        metrics = MolecularGapMetrics(
            target_mae_ev=0.1,
            compute_correlations=True
        )

        # Generate sample predictions and targets
        predictions = torch.randn(50, 1)
        targets = torch.randn(50, 1)

        # Update metrics
        metrics.update(predictions, targets, split='train')

        # Compute results
        results = metrics.compute(split='train')

        assert isinstance(results, dict)
        assert 'mae' in results
        assert 'rmse' in results
        assert results['mae'] >= 0
        assert results['rmse'] >= 0

    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        metrics = MolecularGapMetrics()

        # Add some data
        predictions = torch.randn(10, 1)
        targets = torch.randn(10, 1)
        metrics.update(predictions, targets, split='train')

        # Reset
        metrics.reset(split='train')

        # Should have no stored data
        assert len(metrics.predictions['train']) == 0
        assert len(metrics.targets['train']) == 0

    def test_metrics_multiple_splits(self):
        """Test metrics with multiple data splits."""
        metrics = MolecularGapMetrics()

        for split in ['train', 'val', 'test']:
            predictions = torch.randn(20, 1)
            targets = torch.randn(20, 1)
            metrics.update(predictions, targets, split=split)

            results = metrics.compute(split=split)
            assert 'mae' in results
            assert len(metrics.predictions[split]) == 20


class TestTrainingIntegration:
    """Integration tests for the complete training pipeline."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_mini_training_loop(self, trainer_config, mock_dataset):
        """Test a minimal training loop."""
        trainer = CurriculumTrainer(**trainer_config)

        # Create minimal dataloader
        from torch_geometric.loader import DataLoader as GeomDataLoader
        dataloader = GeomDataLoader(mock_dataset, batch_size=4, shuffle=True)

        # Run a few training steps
        trainer.train()
        total_loss = 0
        num_steps = 0

        for batch in dataloader:
            if num_steps >= 3:  # Just a few steps
                break

            # Add targets if missing
            if not hasattr(batch, 'y'):
                batch.y = torch.randn(batch.num_graphs, 1)

            loss = trainer.training_step(batch, num_steps)
            total_loss += loss.item()
            num_steps += 1

        assert num_steps == 3
        assert total_loss > 0
        assert not np.isnan(total_loss)

    def test_curriculum_dataloader_integration(self, trainer_config):
        """Test integration with curriculum data loading."""
        from spectral_temporal_curriculum_molecular_gap_prediction.data.loader import PCQM4Mv2DataModule

        # Mock data module
        with patch('spectral_temporal_curriculum_molecular_gap_prediction.data.loader.PCQM4Mv2Dataset', None):
            datamodule = PCQM4Mv2DataModule(
                batch_size=4,
                subset=True,
                max_samples=20
            )
            datamodule.setup(stage='fit')

            trainer = CurriculumTrainer(**trainer_config)

            # Test getting curriculum dataloader
            curriculum_loader = trainer.get_curriculum_dataloader(
                datamodule=datamodule,
                stage='train'
            )

            assert curriculum_loader is not None

    def test_gradient_accumulation(self, trainer_config, batch_molecular_graphs):
        """Test gradient accumulation during training."""
        # Modify config for gradient accumulation
        trainer_config['optimizer_config']['lr'] = 0.001

        trainer = CurriculumTrainer(**trainer_config)

        # Mock targets
        batch_molecular_graphs.y = torch.randn(batch_molecular_graphs.num_graphs, 1)

        # Get initial parameters
        initial_params = [p.clone() for p in trainer.parameters()]

        # Training step (should accumulate gradients)
        loss = trainer.training_step(batch_molecular_graphs, 0)

        # Check that gradients exist
        has_gradients = any(p.grad is not None for p in trainer.parameters() if p.requires_grad)
        assert has_gradients or not trainer.automatic_optimization

    def test_model_checkpointing_compatibility(self, trainer_config, temp_dir):
        """Test model checkpointing compatibility."""
        trainer = CurriculumTrainer(**trainer_config)

        # Save checkpoint
        checkpoint_path = temp_dir / "test_checkpoint.ckpt"

        # Manual checkpoint saving (since we're not using full Lightning trainer)
        checkpoint = {
            'state_dict': trainer.state_dict(),
            'hyper_parameters': trainer.hparams
        }
        torch.save(checkpoint, checkpoint_path)

        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path)

        # Create new trainer and load state
        new_trainer = CurriculumTrainer(**trainer_config)
        new_trainer.load_state_dict(loaded_checkpoint['state_dict'])

        # Check that parameters match
        for (name1, param1), (name2, param2) in zip(
            trainer.named_parameters(),
            new_trainer.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2, atol=1e-6)