"""Comprehensive tests for data preprocessing modules."""

import unittest
import sys
import logging
import tempfile
from pathlib import Path
import torch
import numpy as np
from torch_geometric.data import Data

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Disable debug logging for cleaner test output
logging.getLogger().setLevel(logging.WARNING)

try:
    from spectral_temporal_curriculum_molecular_gap_prediction.data.preprocessing import (
        SpectralFeatureExtractor, MolecularGraphProcessor
    )
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    PREPROCESSING_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestSpectralFeatureExtractor(unittest.TestCase):
    """Test cases for SpectralFeatureExtractor class."""

    def setUp(self):
        """Set up test fixtures."""
        if not PREPROCESSING_AVAILABLE:
            self.skipTest(f"Preprocessing imports failed: {IMPORT_ERROR}")

        self.extractor = SpectralFeatureExtractor(
            k_eigenvalues=5,
            chebyshev_order_max=10,
            spectral_tolerance=1e-3
        )

    def test_initialization_valid_params(self):
        """Test extractor initialization with valid parameters."""
        extractor = SpectralFeatureExtractor(
            k_eigenvalues=8,
            chebyshev_order_max=15,
            spectral_tolerance=1e-4
        )

        self.assertEqual(extractor.k_eigenvalues, 8)
        self.assertEqual(extractor.chebyshev_order_max, 15)
        self.assertEqual(extractor.spectral_tolerance, 1e-4)

    def test_initialization_invalid_params(self):
        """Test extractor initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            SpectralFeatureExtractor(k_eigenvalues=0)

        with self.assertRaises(ValueError):
            SpectralFeatureExtractor(chebyshev_order_max=-1)

        with self.assertRaises(ValueError):
            SpectralFeatureExtractor(spectral_tolerance=0)

        with self.assertRaises(ValueError):
            SpectralFeatureExtractor(spectral_tolerance=-0.1)

    def test_extract_features_simple_graph(self):
        """Test feature extraction on a simple molecular graph."""
        # Create a simple molecular graph (triangle)
        x = torch.randn(3, 9)  # 3 atoms with 9 features each
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        features = self.extractor.extract_spectral_features(data)

        # Check that all expected features are present
        expected_keys = {
            'spectral_gap', 'algebraic_connectivity', 'chebyshev_order',
            'num_nodes', 'num_edges', 'density', 'spectral_complexity'
        }
        self.assertEqual(set(features.keys()), expected_keys)

        # Check feature value ranges
        self.assertGreaterEqual(features['spectral_gap'], 0.0)
        self.assertGreaterEqual(features['algebraic_connectivity'], 0.0)
        self.assertGreaterEqual(features['chebyshev_order'], 1.0)
        self.assertEqual(features['num_nodes'], 3.0)
        self.assertEqual(features['num_edges'], 3.0)
        self.assertGreaterEqual(features['density'], 0.0)
        self.assertLessEqual(features['density'], 1.0)
        self.assertGreaterEqual(features['spectral_complexity'], 0.0)
        self.assertLessEqual(features['spectral_complexity'], 1.0)

    def test_extract_features_linear_graph(self):
        """Test feature extraction on a linear molecular graph."""
        # Create a linear chain graph
        x = torch.randn(5, 9)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, num_nodes=5)

        features = self.extractor.extract_spectral_features(data)

        self.assertEqual(features['num_nodes'], 5.0)
        self.assertEqual(features['num_edges'], 4.0)
        # Linear graphs should have lower density than complete graphs
        self.assertLess(features['density'], 0.5)

    def test_extract_features_complete_graph(self):
        """Test feature extraction on a complete molecular graph."""
        # Create a complete graph (all atoms bonded)
        num_nodes = 4
        x = torch.randn(num_nodes, 9)
        # Complete graph edges
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edges.extend([[i, j], [j, i]])
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

        features = self.extractor.extract_spectral_features(data)

        self.assertEqual(features['num_nodes'], float(num_nodes))
        # Complete graph should have high density
        self.assertGreater(features['density'], 0.8)

    def test_extract_features_empty_graph(self):
        """Test feature extraction on an empty graph."""
        x = torch.empty(0, 9)
        edge_index = torch.empty(2, 0, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, num_nodes=0)

        features = self.extractor.extract_spectral_features(data)

        # Should return default features for empty graph
        expected_features = self.extractor._default_features()
        self.assertEqual(features, expected_features)

    def test_extract_features_single_node(self):
        """Test feature extraction on a single isolated node."""
        x = torch.randn(1, 9)
        edge_index = torch.empty(2, 0, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, num_nodes=1)

        features = self.extractor.extract_spectral_features(data)

        self.assertEqual(features['num_nodes'], 1.0)
        self.assertEqual(features['num_edges'], 0.0)
        self.assertEqual(features['density'], 0.0)

    def test_extract_features_invalid_data(self):
        """Test feature extraction with invalid data."""
        # Missing edge_index
        data = Data(x=torch.randn(5, 9), num_nodes=5)
        features = self.extractor.extract_spectral_features(data)
        expected_features = self.extractor._default_features()
        self.assertEqual(features, expected_features)

        # Invalid num_nodes
        data = Data(
            x=torch.randn(5, 9),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            num_nodes=0
        )
        features = self.extractor.extract_spectral_features(data)
        self.assertEqual(features, expected_features)

    def test_default_features(self):
        """Test that default features have correct structure."""
        default_features = self.extractor._default_features()

        expected_keys = {
            'spectral_gap', 'algebraic_connectivity', 'chebyshev_order',
            'num_nodes', 'num_edges', 'density', 'spectral_complexity'
        }
        self.assertEqual(set(default_features.keys()), expected_keys)

        # All default values should be valid
        for key, value in default_features.items():
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0.0)

    def test_compute_eigenvalues_small_graph(self):
        """Test eigenvalue computation on small graphs."""
        # Create a small symmetric matrix (Laplacian-like)
        import scipy.sparse as sp
        L = sp.coo_matrix([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=float)

        eigenvalues = self.extractor._compute_eigenvalues(L, 3)

        # Should have computed eigenvalues
        self.assertGreater(len(eigenvalues), 0)
        self.assertLessEqual(len(eigenvalues), 3)
        # Eigenvalues should be sorted
        self.assertTrue(np.all(eigenvalues[:-1] <= eigenvalues[1:]))

    def test_complexity_score_calculation(self):
        """Test complexity score calculation."""
        features = {
            'chebyshev_order': 10.0,
            'num_nodes': 20.0,
            'density': 0.3,
            'spectral_gap': 0.5
        }

        complexity = self.extractor._compute_complexity_score(features)

        self.assertIsInstance(complexity, float)
        self.assertGreaterEqual(complexity, 0.0)
        self.assertLessEqual(complexity, 1.0)

    def test_chebyshev_order_estimation(self):
        """Test Chebyshev polynomial order estimation."""
        # Test with different eigenvalue ranges
        eigenvalues_narrow = np.array([0.0, 0.1, 0.2])
        order_narrow = self.extractor._estimate_chebyshev_order(eigenvalues_narrow)

        eigenvalues_wide = np.array([0.0, 1.0, 2.0, 3.0])
        order_wide = self.extractor._estimate_chebyshev_order(eigenvalues_wide)

        # Wider spectral range should require higher order
        self.assertGreaterEqual(order_wide, order_narrow)
        self.assertLessEqual(order_wide, self.extractor.chebyshev_order_max)


class TestMolecularGraphProcessor(unittest.TestCase):
    """Test cases for MolecularGraphProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        if not PREPROCESSING_AVAILABLE:
            self.skipTest(f"Preprocessing imports failed: {IMPORT_ERROR}")

        self.processor = MolecularGraphProcessor(
            node_feature_dim=64,
            edge_feature_dim=32,
            add_self_loops=True,
            normalize_features=True
        )

    def test_initialization_valid_params(self):
        """Test processor initialization with valid parameters."""
        processor = MolecularGraphProcessor(
            node_feature_dim=128,
            edge_feature_dim=64,
            add_self_loops=False,
            normalize_features=False
        )

        self.assertEqual(processor.node_feature_dim, 128)
        self.assertEqual(processor.edge_feature_dim, 64)
        self.assertFalse(processor.add_self_loops)
        self.assertFalse(processor.normalize_features)

    def test_process_graph_basic(self):
        """Test basic graph processing functionality."""
        # Create test molecular graph
        x = torch.randn(5, 10)  # 5 atoms, 10 features each
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_attr = torch.randn(3, 5)  # 3 bonds, 5 features each
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=5)

        processed_data, spectral_features = self.processor.process_graph(data)

        # Check processed data structure
        self.assertIsInstance(processed_data, Data)
        self.assertIsInstance(spectral_features, dict)

        # Check that spectral features are added as attributes
        self.assertTrue(hasattr(processed_data, 'spectral_num_nodes'))
        self.assertTrue(hasattr(processed_data, 'spectral_complexity'))

        # Check feature dimensions are correct
        self.assertEqual(processed_data.x.size(1), self.processor.node_feature_dim)
        if processed_data.edge_attr is not None:
            self.assertEqual(processed_data.edge_attr.size(1), self.processor.edge_feature_dim)

    def test_process_graph_with_self_loops(self):
        """Test graph processing with self-loop addition."""
        processor = MolecularGraphProcessor(add_self_loops=True)

        x = torch.randn(4, 10)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_attr = torch.randn(2, 5)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=4)

        original_num_edges = edge_index.size(1)
        processed_data, _ = processor.process_graph(data)

        # Should have added self-loops
        expected_num_edges = original_num_edges + data.num_nodes
        self.assertEqual(processed_data.edge_index.size(1), expected_num_edges)

        # Self-loops should be present
        for i in range(data.num_nodes):
            self_loop_exists = torch.any(
                (processed_data.edge_index[0] == i) &
                (processed_data.edge_index[1] == i)
            )
            self.assertTrue(self_loop_exists, f"Self-loop missing for node {i}")

    def test_process_graph_without_self_loops(self):
        """Test graph processing without self-loop addition."""
        processor = MolecularGraphProcessor(add_self_loops=False)

        x = torch.randn(4, 10)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, num_nodes=4)

        original_num_edges = edge_index.size(1)
        processed_data, _ = processor.process_graph(data)

        # Should not have added self-loops
        self.assertEqual(processed_data.edge_index.size(1), original_num_edges)

    def test_feature_resizing_padding(self):
        """Test feature resizing with padding."""
        # Test when input features are smaller than target
        small_features = torch.randn(5, 10)  # Smaller than target 64
        resized = self.processor._resize_features(small_features, 64)

        self.assertEqual(resized.shape, (5, 64))
        # Original features should be preserved
        self.assertTrue(torch.allclose(resized[:, :10], small_features))
        # Padding should be zero
        self.assertTrue(torch.allclose(resized[:, 10:], torch.zeros(5, 54)))

    def test_feature_resizing_truncation(self):
        """Test feature resizing with truncation."""
        # Test when input features are larger than target
        large_features = torch.randn(5, 128)  # Larger than target 64
        resized = self.processor._resize_features(large_features, 64)

        self.assertEqual(resized.shape, (5, 64))
        # Should be truncated version of original
        self.assertTrue(torch.allclose(resized, large_features[:, :64]))

    def test_feature_resizing_exact_size(self):
        """Test feature resizing when size is already correct."""
        correct_features = torch.randn(5, 64)  # Exactly target size
        resized = self.processor._resize_features(correct_features, 64)

        self.assertEqual(resized.shape, (5, 64))
        # Should be unchanged
        self.assertTrue(torch.allclose(resized, correct_features))

    def test_fit_normalizer(self):
        """Test feature normalization fitting."""
        # Create training data with known statistics
        data_list = []
        for _ in range(10):
            x = torch.randn(5, 10) * 2.0 + 1.0  # mean=1, std=2
            edge_attr = torch.randn(3, 5) * 0.5 + 0.5  # mean=0.5, std=0.5
            data = Data(x=x, edge_attr=edge_attr)
            data_list.append(data)

        self.processor.fit_normalizer(data_list)

        # Check that normalization statistics are computed
        self.assertIsNotNone(self.processor.node_mean)
        self.assertIsNotNone(self.processor.node_std)
        self.assertIsNotNone(self.processor.edge_mean)
        self.assertIsNotNone(self.processor.edge_std)

        # Statistics should be reasonable
        self.assertLess(abs(self.processor.node_mean.mean().item() - 1.0), 0.3)
        self.assertLess(abs(self.processor.edge_mean.mean().item() - 0.5), 0.2)

    def test_feature_normalization_application(self):
        """Test that feature normalization is applied correctly."""
        # Create processor with normalization
        processor = MolecularGraphProcessor(normalize_features=True)

        # Fit normalizer with known data
        x_train = torch.ones(10, 5) * 2.0  # All features = 2.0
        data_train = Data(x=x_train)
        processor.fit_normalizer([data_train])

        # Process new data
        x_test = torch.ones(3, 5) * 2.0  # Same distribution
        data_test = Data(x=x_test, edge_index=torch.empty(2, 0, dtype=torch.long), num_nodes=3)

        processed_data, _ = processor.process_graph(data_test)

        # After normalization, features should be close to zero (standardized)
        normalized_features = processed_data.x[:, :5]  # Original feature part
        self.assertLess(abs(normalized_features.mean().item()), 0.1)

    def test_no_normalization(self):
        """Test processing without feature normalization."""
        processor = MolecularGraphProcessor(normalize_features=False)

        x = torch.randn(5, 10)
        data = Data(x=x, edge_index=torch.empty(2, 0, dtype=torch.long), num_nodes=5)

        processed_data, _ = processor.process_graph(data)

        # Features should be resized but not normalized
        original_resized = processor._resize_features(x, processor.node_feature_dim)
        self.assertTrue(torch.allclose(processed_data.x, original_resized))


class TestProcessingIntegration(unittest.TestCase):
    """Integration tests for preprocessing components."""

    def setUp(self):
        """Set up test fixtures."""
        if not PREPROCESSING_AVAILABLE:
            self.skipTest(f"Preprocessing imports failed: {IMPORT_ERROR}")

    def test_full_processing_pipeline(self):
        """Test complete processing pipeline from raw to processed data."""
        processor = MolecularGraphProcessor(
            node_feature_dim=64,
            edge_feature_dim=32,
            add_self_loops=True,
            normalize_features=True
        )

        # Create diverse training data for normalization
        training_data = []
        for i in range(20):
            num_nodes = np.random.randint(3, 10)
            num_edges = np.random.randint(2, num_nodes * 2)

            x = torch.randn(num_nodes, np.random.randint(5, 15))
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            edge_attr = torch.randn(num_edges, np.random.randint(2, 8))

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
            training_data.append(data)

        # Fit normalizer
        processor.fit_normalizer(training_data)

        # Process test data
        test_x = torch.randn(6, 12)
        test_edge_index = torch.tensor([[0, 1, 2, 1], [1, 2, 0, 3]], dtype=torch.long)
        test_edge_attr = torch.randn(4, 6)
        test_data = Data(
            x=test_x,
            edge_index=test_edge_index,
            edge_attr=test_edge_attr,
            num_nodes=6
        )

        processed_data, spectral_features = processor.process_graph(test_data)

        # Check all processing steps were applied
        self.assertEqual(processed_data.x.shape[1], 64)  # Node features resized
        self.assertEqual(processed_data.edge_attr.shape[1], 32)  # Edge features resized
        self.assertGreater(processed_data.edge_index.shape[1], test_edge_index.shape[1])  # Self-loops added

        # Check spectral features are computed
        self.assertIn('spectral_complexity', spectral_features)
        self.assertIn('num_nodes', spectral_features)

        # Check spectral features are attached to data
        self.assertTrue(hasattr(processed_data, 'spectral_complexity'))

    def test_batch_processing_consistency(self):
        """Test that batch processing gives consistent results."""
        processor = MolecularGraphProcessor()

        # Create test data
        data_list = []
        for _ in range(5):
            x = torch.randn(4, 10)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, num_nodes=4)
            data_list.append(data)

        # Process individually
        individual_results = []
        for data in data_list:
            processed_data, spectral_features = processor.process_graph(data)
            individual_results.append((processed_data, spectral_features))

        # Process in batch (simulate batch processing)
        batch_results = []
        for data in data_list:
            processed_data, spectral_features = processor.process_graph(data)
            batch_results.append((processed_data, spectral_features))

        # Results should be identical
        for (proc1, spec1), (proc2, spec2) in zip(individual_results, batch_results):
            self.assertTrue(torch.allclose(proc1.x, proc2.x))
            self.assertTrue(torch.allclose(proc1.edge_index, proc2.edge_index))
            self.assertEqual(spec1, spec2)

    def test_edge_cases_handling(self):
        """Test handling of various edge cases."""
        processor = MolecularGraphProcessor()

        # Test with disconnected graph
        x = torch.randn(4, 10)
        edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)  # Disconnected
        data = Data(x=x, edge_index=edge_index, num_nodes=4)

        try:
            processed_data, spectral_features = processor.process_graph(data)
            # Should handle disconnected graphs gracefully
            self.assertIsNotNone(processed_data)
            self.assertIsNotNone(spectral_features)
        except Exception as e:
            self.fail(f"Failed to handle disconnected graph: {e}")

        # Test with no edges
        x = torch.randn(3, 10)
        edge_index = torch.empty(2, 0, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        processed_data, spectral_features = processor.process_graph(data)
        self.assertIsNotNone(processed_data)
        self.assertEqual(spectral_features['num_edges'], 0.0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)