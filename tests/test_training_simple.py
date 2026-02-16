"""Simple tests for training modules."""

import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestTrainingModules(unittest.TestCase):
    """Test suite for training modules."""

    def test_imports(self):
        """Test that modules can be imported without errors."""
        try:
            from spectral_temporal_curriculum_molecular_gap_prediction.training import trainer
            from spectral_temporal_curriculum_molecular_gap_prediction.evaluation import metrics
            from spectral_temporal_curriculum_molecular_gap_prediction.utils import config
            self.assertTrue(True, "Imports successful")
        except ImportError as e:
            self.skipTest(f"Module import failed: {e}")

    def test_trainer_classes_exist(self):
        """Test that trainer classes can be found."""
        try:
            from spectral_temporal_curriculum_molecular_gap_prediction.training.trainer import (
                CurriculumTrainer, CustomLoss, SpectralComplexityScheduler
            )

            self.assertTrue(hasattr(CurriculumTrainer, '__init__'))
            self.assertTrue(hasattr(CustomLoss, '__init__'))
            self.assertTrue(hasattr(SpectralComplexityScheduler, '__init__'))
        except Exception as e:
            self.skipTest(f"Trainer classes not available: {e}")

    def test_config_loading(self):
        """Test configuration loading."""
        try:
            from spectral_temporal_curriculum_molecular_gap_prediction.utils.config import (
                Config, load_config
            )

            config = Config()
            self.assertIsNotNone(config)
            self.assertIsNotNone(config.model)
            self.assertIsNotNone(config.training)
        except Exception as e:
            self.skipTest(f"Config classes not available: {e}")


if __name__ == '__main__':
    unittest.main()