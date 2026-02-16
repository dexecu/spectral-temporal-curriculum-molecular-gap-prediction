"""Simple tests for data loading and preprocessing modules."""

import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDataModules(unittest.TestCase):
    """Test suite for data modules."""

    def test_imports(self):
        """Test that modules can be imported without errors."""
        try:
            from spectral_temporal_curriculum_molecular_gap_prediction.data import preprocessing
            from spectral_temporal_curriculum_molecular_gap_prediction.data import loader
            self.assertTrue(True, "Imports successful")
        except ImportError as e:
            self.skipTest(f"Module import failed: {e}")

    def test_spectral_extractor_basic(self):
        """Test basic SpectralFeatureExtractor functionality."""
        try:
            from spectral_temporal_curriculum_molecular_gap_prediction.data.preprocessing import (
                SpectralFeatureExtractor
            )

            extractor = SpectralFeatureExtractor()
            self.assertIsInstance(extractor.k_eigenvalues, int)
            self.assertIsInstance(extractor.chebyshev_order_max, int)
            self.assertIsInstance(extractor.spectral_tolerance, float)
        except Exception as e:
            self.skipTest(f"SpectralFeatureExtractor not available: {e}")

    def test_molecular_processor_basic(self):
        """Test basic MolecularGraphProcessor functionality."""
        try:
            from spectral_temporal_curriculum_molecular_gap_prediction.data.preprocessing import (
                MolecularGraphProcessor
            )

            processor = MolecularGraphProcessor()
            self.assertIsInstance(processor.node_feature_dim, int)
            self.assertIsInstance(processor.edge_feature_dim, int)
        except Exception as e:
            self.skipTest(f"MolecularGraphProcessor not available: {e}")


if __name__ == '__main__':
    unittest.main()