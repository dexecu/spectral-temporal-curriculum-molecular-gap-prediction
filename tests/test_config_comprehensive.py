"""Comprehensive tests for configuration management."""

import unittest
import sys
import tempfile
import os
from pathlib import Path
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from spectral_temporal_curriculum_molecular_gap_prediction.utils.config import (
        Config, ModelConfig, OptimizerConfig, SchedulerConfig, CurriculumConfig,
        LossConfig, EvaluationConfig, DataConfig, TrainingConfig, LoggingConfig,
        ExperimentConfig, load_config, save_config, merge_configs, validate_config,
        create_default_config_file, override_config_from_args, _deep_merge_dicts
    )
    CONFIG_AVAILABLE = True
except ImportError as e:
    CONFIG_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestConfigDataClasses(unittest.TestCase):
    """Test individual configuration dataclasses."""

    def setUp(self):
        """Set up test fixtures."""
        if not CONFIG_AVAILABLE:
            self.skipTest(f"Config imports failed: {IMPORT_ERROR}")

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()

        self.assertEqual(config.node_features, 128)
        self.assertEqual(config.edge_features, 64)
        self.assertEqual(config.hidden_dim, 256)
        self.assertEqual(config.mp_layers, 4)
        self.assertEqual(config.num_spectral_filters, 6)
        self.assertEqual(config.max_chebyshev_order, 20)
        self.assertEqual(config.fusion_type, 'cross_attention')
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.pooling, 'attention')
        self.assertEqual(config.output_dim, 1)

    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            hidden_dim=512,
            mp_layers=6,
            fusion_type='concat',
            dropout=0.2
        )

        self.assertEqual(config.hidden_dim, 512)
        self.assertEqual(config.mp_layers, 6)
        self.assertEqual(config.fusion_type, 'concat')
        self.assertEqual(config.dropout, 0.2)
        # Should keep defaults for unspecified values
        self.assertEqual(config.node_features, 128)

    def test_optimizer_config_defaults(self):
        """Test OptimizerConfig default values."""
        config = OptimizerConfig()

        self.assertEqual(config.name, 'adamw')
        self.assertEqual(config.lr, 1e-3)
        self.assertEqual(config.weight_decay, 0.01)
        self.assertEqual(config.betas, (0.9, 0.999))
        self.assertEqual(config.momentum, 0.9)

    def test_curriculum_config_defaults(self):
        """Test CurriculumConfig default values."""
        config = CurriculumConfig()

        self.assertEqual(config.initial_fraction, 0.1)
        self.assertEqual(config.final_fraction, 1.0)
        self.assertEqual(config.warmup_epochs, 5)
        self.assertEqual(config.total_epochs, 100)
        self.assertEqual(config.growth_strategy, 'exponential')
        self.assertEqual(config.min_growth_rate, 0.05)

    def test_evaluation_config_defaults(self):
        """Test EvaluationConfig default values."""
        config = EvaluationConfig()

        self.assertEqual(config.target_mae_ev, 0.082)
        self.assertEqual(config.convergence_window, 10)
        self.assertEqual(config.tail_percentile, 95.0)
        self.assertTrue(config.compute_correlations)
        self.assertTrue(config.track_convergence)


class TestMainConfig(unittest.TestCase):
    """Test the main Config class."""

    def setUp(self):
        """Set up test fixtures."""
        if not CONFIG_AVAILABLE:
            self.skipTest(f"Config imports failed: {IMPORT_ERROR}")

    def test_config_initialization_defaults(self):
        """Test Config initialization with all defaults."""
        config = Config()

        # Check that all sub-configs are initialized
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.optimizer, OptimizerConfig)
        self.assertIsInstance(config.scheduler, SchedulerConfig)
        self.assertIsInstance(config.curriculum, CurriculumConfig)
        self.assertIsInstance(config.loss, LossConfig)
        self.assertIsInstance(config.evaluation, EvaluationConfig)
        self.assertIsInstance(config.data, DataConfig)
        self.assertIsInstance(config.training, TrainingConfig)
        self.assertIsInstance(config.logging, LoggingConfig)
        self.assertIsInstance(config.experiment, ExperimentConfig)

        # Check global settings
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.device, 'auto')
        self.assertEqual(config.num_gpus, 1)
        self.assertEqual(config.strategy, 'auto')

    def test_config_validation_valid(self):
        """Test config validation with valid parameters."""
        config = Config()
        # Should not raise any exceptions
        config._validate_config()

    def test_config_validation_invalid_model(self):
        """Test config validation with invalid model parameters."""
        config = Config()
        config.model.hidden_dim = 0  # Invalid

        with self.assertRaises(AssertionError):
            config._validate_config()

    def test_config_validation_invalid_curriculum(self):
        """Test config validation with invalid curriculum parameters."""
        config = Config()
        config.curriculum.initial_fraction = 1.5  # Invalid (> 1.0)

        with self.assertRaises(AssertionError):
            config._validate_config()

        config = Config()
        config.curriculum.growth_strategy = 'invalid'  # Invalid strategy

        with self.assertRaises(AssertionError):
            config._validate_config()

    def test_config_validation_invalid_data(self):
        """Test config validation with invalid data parameters."""
        config = Config()
        config.data.batch_size = -1  # Invalid

        with self.assertRaises(AssertionError):
            config._validate_config()

    def test_config_to_dict(self):
        """Test config conversion to dictionary."""
        config = Config()
        config_dict = config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertIn('model', config_dict)
        self.assertIn('optimizer', config_dict)
        self.assertIn('seed', config_dict)

        # Check nested structure
        self.assertIsInstance(config_dict['model'], dict)
        self.assertIn('hidden_dim', config_dict['model'])

    def test_config_from_dict(self):
        """Test config creation from dictionary."""
        config_dict = {
            'model': {'hidden_dim': 512, 'dropout': 0.2},
            'optimizer': {'lr': 5e-4},
            'seed': 123
        }

        config = Config.from_dict(config_dict)

        self.assertEqual(config.model.hidden_dim, 512)
        self.assertEqual(config.model.dropout, 0.2)
        self.assertEqual(config.optimizer.lr, 5e-4)
        self.assertEqual(config.seed, 123)
        # Should keep defaults for unspecified values
        self.assertEqual(config.model.node_features, 128)

    def test_device_setup_auto_cpu(self):
        """Test automatic device setup for CPU."""
        # Remove CUDA environment variable if present
        original_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']

        try:
            config = Config(device='auto', num_gpus=1)
            config._setup_device()

            self.assertEqual(config.device, 'cpu')
            self.assertEqual(config.num_gpus, 0)
        finally:
            # Restore original environment
            if original_cuda is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda

    def test_device_setup_auto_gpu(self):
        """Test automatic device setup for GPU."""
        # Set CUDA environment variable
        original_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        try:
            config = Config(device='auto', num_gpus=1)
            config._setup_device()

            self.assertEqual(config.device, 'gpu')
            self.assertEqual(config.num_gpus, 1)
        finally:
            # Restore original environment
            if original_cuda is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda
            else:
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']


class TestConfigFileOperations(unittest.TestCase):
    """Test configuration file loading and saving."""

    def setUp(self):
        """Set up test fixtures."""
        if not CONFIG_AVAILABLE:
            self.skipTest(f"Config imports failed: {IMPORT_ERROR}")

        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_save_and_load_config(self):
        """Test saving and loading configuration files."""
        config = Config()
        config.model.hidden_dim = 512
        config.optimizer.lr = 5e-4
        config.seed = 123

        # Save config
        config_file = self.temp_path / "test_config.yaml"
        save_config(config, config_file)

        # Check file was created
        self.assertTrue(config_file.exists())

        # Load config
        loaded_config = load_config(config_file)

        # Check values are preserved
        self.assertEqual(loaded_config.model.hidden_dim, 512)
        self.assertEqual(loaded_config.optimizer.lr, 5e-4)
        self.assertEqual(loaded_config.seed, 123)

    def test_load_nonexistent_config(self):
        """Test loading nonexistent configuration file."""
        nonexistent_file = self.temp_path / "nonexistent.yaml"
        config = load_config(nonexistent_file)

        # Should return default config
        self.assertIsInstance(config, Config)
        self.assertEqual(config.model.hidden_dim, 256)  # Default value

    def test_load_empty_config_file(self):
        """Test loading empty configuration file."""
        empty_file = self.temp_path / "empty.yaml"
        empty_file.write_text("")

        config = load_config(empty_file)

        # Should return default config
        self.assertIsInstance(config, Config)
        self.assertEqual(config.model.hidden_dim, 256)

    def test_load_malformed_config_file(self):
        """Test loading malformed configuration file."""
        malformed_file = self.temp_path / "malformed.yaml"
        malformed_file.write_text("invalid: yaml: content: [")

        config = load_config(malformed_file)

        # Should return default config on error
        self.assertIsInstance(config, Config)
        self.assertEqual(config.model.hidden_dim, 256)

    def test_save_config_creates_directory(self):
        """Test that save_config creates directories if needed."""
        config = Config()
        nested_file = self.temp_path / "nested" / "dir" / "config.yaml"

        save_config(config, nested_file)

        self.assertTrue(nested_file.exists())
        self.assertTrue(nested_file.parent.exists())

    def test_create_default_config_file(self):
        """Test creation of default configuration file."""
        default_file = self.temp_path / "default.yaml"
        create_default_config_file(default_file)

        self.assertTrue(default_file.exists())

        # Load and verify it's valid
        config = load_config(default_file)
        self.assertIsInstance(config, Config)

    def test_config_file_format(self):
        """Test that saved config file has correct YAML format."""
        config = Config()
        config.model.hidden_dim = 512

        config_file = self.temp_path / "format_test.yaml"
        save_config(config, config_file)

        # Read raw YAML and verify structure
        with open(config_file, 'r') as f:
            yaml_content = yaml.safe_load(f)

        self.assertIn('model', yaml_content)
        self.assertIn('hidden_dim', yaml_content['model'])
        self.assertEqual(yaml_content['model']['hidden_dim'], 512)


class TestConfigMerging(unittest.TestCase):
    """Test configuration merging functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not CONFIG_AVAILABLE:
            self.skipTest(f"Config imports failed: {IMPORT_ERROR}")

    def test_merge_configs_simple(self):
        """Test simple config merging."""
        base_config = Config()
        override_dict = {'model': {'hidden_dim': 512}, 'seed': 999}

        merged_config = merge_configs(base_config, override_dict)

        self.assertEqual(merged_config.model.hidden_dim, 512)
        self.assertEqual(merged_config.seed, 999)
        # Should preserve unspecified values
        self.assertEqual(merged_config.model.dropout, 0.1)

    def test_merge_configs_nested(self):
        """Test nested config merging."""
        base_config = Config()
        override_dict = {
            'model': {'hidden_dim': 512, 'dropout': 0.2},
            'optimizer': {'lr': 1e-4},
            'training': {'max_epochs': 200}
        }

        merged_config = merge_configs(base_config, override_dict)

        self.assertEqual(merged_config.model.hidden_dim, 512)
        self.assertEqual(merged_config.model.dropout, 0.2)
        self.assertEqual(merged_config.optimizer.lr, 1e-4)
        self.assertEqual(merged_config.training.max_epochs, 200)
        # Should preserve other values
        self.assertEqual(merged_config.model.node_features, 128)

    def test_deep_merge_dicts(self):
        """Test deep dictionary merging utility."""
        dict1 = {
            'a': {'x': 1, 'y': 2},
            'b': 3,
            'c': {'z': 4}
        }
        dict2 = {
            'a': {'y': 20, 'w': 30},
            'b': 30,
            'd': 40
        }

        merged = _deep_merge_dicts(dict1, dict2)

        expected = {
            'a': {'x': 1, 'y': 20, 'w': 30},
            'b': 30,
            'c': {'z': 4},
            'd': 40
        }

        self.assertEqual(merged, expected)

    def test_override_config_from_args(self):
        """Test config override from command-line arguments."""
        config = Config()
        args = {
            'learning_rate': 5e-4,
            'batch_size': 128,
            'max_epochs': 150,
            'hidden_dim': 512,
            'dropout': 0.15,
            'seed': 42
        }

        overridden_config = override_config_from_args(config, args)

        self.assertEqual(overridden_config.optimizer.lr, 5e-4)
        self.assertEqual(overridden_config.data.batch_size, 128)
        self.assertEqual(overridden_config.training.max_epochs, 150)
        self.assertEqual(overridden_config.model.hidden_dim, 512)
        self.assertEqual(overridden_config.model.dropout, 0.15)
        self.assertEqual(overridden_config.seed, 42)

    def test_override_config_from_args_partial(self):
        """Test partial config override from arguments."""
        config = Config()
        original_lr = config.optimizer.lr

        args = {'batch_size': 256}  # Only override batch size

        overridden_config = override_config_from_args(config, args)

        self.assertEqual(overridden_config.data.batch_size, 256)
        self.assertEqual(overridden_config.optimizer.lr, original_lr)  # Unchanged

    def test_override_config_unknown_args(self):
        """Test config override with unknown arguments."""
        config = Config()
        args = {
            'learning_rate': 5e-4,
            'unknown_arg': 'should_be_ignored'
        }

        overridden_config = override_config_from_args(config, args)

        # Known arg should be applied
        self.assertEqual(overridden_config.optimizer.lr, 5e-4)
        # Unknown arg should be ignored (no error)
        self.assertIsInstance(overridden_config, Config)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation functions."""

    def setUp(self):
        """Set up test fixtures."""
        if not CONFIG_AVAILABLE:
            self.skipTest(f"Config imports failed: {IMPORT_ERROR}")

    def test_validate_config_valid(self):
        """Test validation of valid configuration."""
        config = Config()
        is_valid = validate_config(config)
        self.assertTrue(is_valid)

    def test_validate_config_invalid(self):
        """Test validation of invalid configuration."""
        config = Config()
        config.model.hidden_dim = 0  # Invalid

        is_valid = validate_config(config)
        self.assertFalse(is_valid)

    def test_validate_config_edge_cases(self):
        """Test validation of edge case configurations."""
        # Test minimum valid values
        config = Config()
        config.model.hidden_dim = 1  # Minimum valid
        config.curriculum.initial_fraction = 0.001  # Very small but valid
        config.curriculum.final_fraction = 1.0  # Maximum valid

        is_valid = validate_config(config)
        self.assertTrue(is_valid)

        # Test boundary invalid values
        config.curriculum.initial_fraction = 0.0  # Invalid (must be > 0)
        is_valid = validate_config(config)
        self.assertFalse(is_valid)


class TestConfigEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        if not CONFIG_AVAILABLE:
            self.skipTest(f"Config imports failed: {IMPORT_ERROR}")

        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_config_with_none_values(self):
        """Test config handling with None values in dictionary."""
        config_dict = {
            'model': {'hidden_dim': None},  # None value
            'optimizer': None,  # None sub-config
        }

        # Should handle None values gracefully
        config = Config.from_dict(config_dict)
        self.assertIsInstance(config, Config)

    def test_save_config_permission_error(self):
        """Test save config with permission error."""
        config = Config()

        # Try to save to a read-only location (this might not work on all systems)
        readonly_file = self.temp_path / "readonly.yaml"
        readonly_file.write_text("test")
        readonly_file.chmod(0o444)  # Read-only

        try:
            # This should raise an exception
            with self.assertRaises(Exception):
                save_config(config, readonly_file)
        except PermissionError:
            pass  # Expected on some systems
        finally:
            # Clean up
            readonly_file.chmod(0o644)

    def test_config_roundtrip_fidelity(self):
        """Test that save/load preserves all configuration values."""
        # Create config with non-default values
        original_config = Config()
        original_config.model.hidden_dim = 512
        original_config.model.dropout = 0.25
        original_config.optimizer.lr = 3e-4
        original_config.curriculum.growth_strategy = 'linear'
        original_config.data.batch_size = 128
        original_config.seed = 12345

        # Save and load
        config_file = self.temp_path / "roundtrip.yaml"
        save_config(original_config, config_file)
        loaded_config = load_config(config_file)

        # Compare all major values
        self.assertEqual(loaded_config.model.hidden_dim, original_config.model.hidden_dim)
        self.assertEqual(loaded_config.model.dropout, original_config.model.dropout)
        self.assertEqual(loaded_config.optimizer.lr, original_config.optimizer.lr)
        self.assertEqual(loaded_config.curriculum.growth_strategy, original_config.curriculum.growth_strategy)
        self.assertEqual(loaded_config.data.batch_size, original_config.data.batch_size)
        self.assertEqual(loaded_config.seed, original_config.seed)

    def test_config_with_list_values(self):
        """Test config handling with list values."""
        config = Config()
        config.experiment.tags = ['test1', 'test2', 'test3']

        # Convert to dict and back
        config_dict = config.to_dict()
        new_config = Config.from_dict(config_dict)

        self.assertEqual(new_config.experiment.tags, ['test1', 'test2', 'test3'])

    def test_config_post_init_validation(self):
        """Test that post-init validation is called during initialization."""
        # This should raise an exception during initialization
        with self.assertRaises(AssertionError):
            Config(
                model=ModelConfig(hidden_dim=0),  # Invalid
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)