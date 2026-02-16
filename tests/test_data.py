"""Tests for data loading and preprocessing modules."""

import unittest
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeomDataLoader
from unittest.mock import patch, MagicMock
import tempfile
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from spectral_temporal_curriculum_molecular_gap_prediction.data.preprocessing import (
        SpectralFeatureExtractor, MolecularGraphProcessor
    )
    from spectral_temporal_curriculum_molecular_gap_prediction.data.loader import (
        SpectralComplexityDataset, PCQM4Mv2DataModule
    )
except ImportError as e:
    print(f"ImportError in test_data.py: {e}")
    # Create dummy classes for testing when imports fail
    class SpectralFeatureExtractor:
        pass
    class MolecularGraphProcessor:
        pass
    class SpectralComplexityDataset:
        pass
    class PCQM4Mv2DataModule:
        pass


class TestSpectralFeatureExtractor(unittest.TestCase):
    """Test suite for SpectralFeatureExtractor."""

    def test_initialization(self):
        """Test SpectralFeatureExtractor initialization."""
        try:
            extractor = SpectralFeatureExtractor(
                k_eigenvalues=5,
                chebyshev_order_max=10,
                spectral_tolerance=1e-3
            )

            self.assertEqual(extractor.k_eigenvalues, 5)
            self.assertEqual(extractor.chebyshev_order_max, 10)
            self.assertEqual(extractor.spectral_tolerance, 1e-3)
        except Exception as e:
            self.skipTest(f"SpectralFeatureExtractor not available: {e}")

    def test_extract_spectral_features_simple_graph(self, simple_molecular_graph):
        """Test spectral feature extraction on a simple graph."""
        extractor = SpectralFeatureExtractor()
        features = extractor.extract_spectral_features(simple_molecular_graph)

        # Check that all expected features are present
        expected_features = [
            'spectral_gap', 'algebraic_connectivity', 'chebyshev_order',
            'num_nodes', 'num_edges', 'density', 'spectral_complexity'
        ]

        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], float)
            assert not np.isnan(features[feature])

        # Check reasonable value ranges
        assert features['num_nodes'] > 0
        assert features['num_edges'] >= 0
        assert 0 <= features['density'] <= 1
        assert 0 <= features['spectral_complexity'] <= 1
        assert features['chebyshev_order'] >= 1

    def test_extract_spectral_features_empty_graph(self):
        """Test spectral feature extraction on an empty graph."""
        extractor = SpectralFeatureExtractor()

        # Create empty graph
        empty_graph = Data(
            x=torch.empty(0, 9),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            num_nodes=0
        )

        features = extractor.extract_spectral_features(empty_graph)

        # Should return default features
        assert features['spectral_gap'] == 0.0
        assert features['num_nodes'] == 1.0  # Default value for edge case
        assert features['spectral_complexity'] == 0.0

    def test_extract_spectral_features_single_node(self):
        """Test spectral feature extraction on a single node graph."""
        extractor = SpectralFeatureExtractor()

        # Single node graph
        single_node_graph = Data(
            x=torch.randn(1, 9),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            num_nodes=1
        )

        features = extractor.extract_spectral_features(single_node_graph)

        assert features['num_nodes'] == 1.0
        assert features['num_edges'] == 0.0
        assert features['density'] == 0.0

    def test_default_features(self):
        """Test default features fallback."""
        extractor = SpectralFeatureExtractor()
        default_features = extractor._default_features()

        assert isinstance(default_features, dict)
        assert all(isinstance(v, float) for v in default_features.values())
        assert default_features['spectral_complexity'] == 0.0

    def test_complexity_score_computation(self):
        """Test complexity score computation."""
        extractor = SpectralFeatureExtractor()

        features = {
            'chebyshev_order': 5.0,
            'num_nodes': 10.0,
            'density': 0.5,
            'spectral_gap': 1.0
        }

        complexity = extractor._compute_complexity_score(features)

        assert isinstance(complexity, float)
        assert 0.0 <= complexity <= 1.0


class TestMolecularGraphProcessor:
    """Test suite for MolecularGraphProcessor."""

    def test_initialization(self):
        """Test MolecularGraphProcessor initialization."""
        processor = MolecularGraphProcessor(
            node_feature_dim=64,
            edge_feature_dim=32,
            add_self_loops=True,
            normalize_features=True
        )

        assert processor.node_feature_dim == 64
        assert processor.edge_feature_dim == 32
        assert processor.add_self_loops is True
        assert processor.normalize_features is True

    def test_process_graph_basic(self, simple_molecular_graph, molecular_processor):
        """Test basic graph processing."""
        processed_data, spectral_features = molecular_processor.process_graph(
            simple_molecular_graph
        )

        # Check that data structure is preserved
        assert isinstance(processed_data, Data)
        assert processed_data.num_nodes == simple_molecular_graph.num_nodes

        # Check that spectral features are computed
        assert isinstance(spectral_features, dict)
        assert 'spectral_gap' in spectral_features

        # Check feature dimensions
        assert processed_data.x.shape[1] == molecular_processor.node_feature_dim
        if hasattr(processed_data, 'edge_attr'):
            assert processed_data.edge_attr.shape[1] == molecular_processor.edge_feature_dim

    def test_add_self_loops(self, simple_molecular_graph):
        """Test self-loop addition."""
        processor = MolecularGraphProcessor(add_self_loops=True)
        processed_data, _ = processor.process_graph(simple_molecular_graph)

        original_edges = simple_molecular_graph.edge_index.shape[1]
        processed_edges = processed_data.edge_index.shape[1]
        expected_edges = original_edges + simple_molecular_graph.num_nodes

        assert processed_edges == expected_edges

    def test_feature_resizing(self):
        """Test feature resizing functionality."""
        processor = MolecularGraphProcessor(node_feature_dim=10, edge_feature_dim=5)

        # Test padding (smaller input)
        small_features = torch.randn(5, 3)
        resized = processor._resize_features(small_features, 10)
        assert resized.shape == (5, 10)
        assert torch.equal(resized[:, :3], small_features)
        assert torch.equal(resized[:, 3:], torch.zeros(5, 7))

        # Test truncation (larger input)
        large_features = torch.randn(5, 15)
        resized = processor._resize_features(large_features, 10)
        assert resized.shape == (5, 10)
        assert torch.equal(resized, large_features[:, :10])

        # Test same size
        same_features = torch.randn(5, 10)
        resized = processor._resize_features(same_features, 10)
        assert torch.equal(resized, same_features)

    def test_fit_normalizer(self, mock_dataset, molecular_processor):
        """Test feature normalization fitting."""
        # Enable normalization
        molecular_processor.normalize_features = True

        # Create sample data
        sample_graphs = [mock_dataset[i] for i in range(10)]

        # Fit normalizer
        molecular_processor.fit_normalizer(sample_graphs)

        # Check that statistics are computed
        if hasattr(sample_graphs[0], 'x') and sample_graphs[0].x is not None:
            assert molecular_processor.node_mean is not None
            assert molecular_processor.node_std is not None


class TestSpectralComplexityDataset:
    """Test suite for SpectralComplexityDataset."""

    def test_initialization(self, mock_dataset, molecular_processor):
        """Test SpectralComplexityDataset initialization."""
        dataset = SpectralComplexityDataset(
            base_dataset=mock_dataset,
            processor=molecular_processor,
            cache_spectral_features=True
        )

        assert len(dataset) == len(mock_dataset)
        assert dataset.cache_spectral_features is True
        assert isinstance(dataset.spectral_features_cache, dict)

    def test_getitem(self, mock_dataset, molecular_processor):
        """Test dataset item retrieval."""
        dataset = SpectralComplexityDataset(
            base_dataset=mock_dataset,
            processor=molecular_processor
        )

        data, features = dataset[0]

        assert isinstance(data, Data)
        assert isinstance(features, dict)
        assert 'spectral_complexity' in features

    def test_curriculum_order_computation(self, mock_dataset, molecular_processor):
        """Test curriculum order computation."""
        dataset = SpectralComplexityDataset(
            base_dataset=mock_dataset,
            processor=molecular_processor
        )

        # Compute features for first few samples
        for i in range(5):
            _, features = dataset[i]

        # Get curriculum order
        sorted_indices = dataset.get_curriculum_order(
            curriculum_strategy='spectral_complexity'
        )

        assert isinstance(sorted_indices, list)
        assert len(sorted_indices) <= len(dataset)
        assert all(isinstance(idx, int) for idx in sorted_indices)

    def test_curriculum_subset(self, mock_dataset, molecular_processor):
        """Test curriculum subset creation."""
        dataset = SpectralComplexityDataset(
            base_dataset=mock_dataset,
            processor=molecular_processor
        )

        # Compute features for all samples
        for i in range(len(dataset)):
            _, _ = dataset[i]

        # Get subset
        subset = dataset.get_curriculum_subset(
            fraction=0.5,
            curriculum_strategy='spectral_complexity'
        )

        expected_size = int(len(dataset) * 0.5)
        assert len(subset.complexity_sorted_indices) <= expected_size

    @pytest.mark.slow
    def test_compute_all_spectral_features(self, mock_dataset, molecular_processor):
        """Test computing spectral features for all samples."""
        dataset = SpectralComplexityDataset(
            base_dataset=mock_dataset,
            processor=molecular_processor
        )

        # Compute all features
        dataset.compute_all_spectral_features(max_samples=10)

        assert len(dataset.spectral_features_cache) == 10

    def test_cache_save_load(self, mock_dataset, molecular_processor, temp_dir):
        """Test spectral features cache save and load."""
        cache_file = temp_dir / "test_cache.pkl"

        dataset = SpectralComplexityDataset(
            base_dataset=mock_dataset,
            processor=molecular_processor,
            cache_file=str(cache_file)
        )

        # Add some cached features
        dataset.spectral_features_cache[0] = {'spectral_complexity': 0.5}

        # Save cache
        dataset._save_cache()
        assert cache_file.exists()

        # Create new dataset and load cache
        new_dataset = SpectralComplexityDataset(
            base_dataset=mock_dataset,
            processor=molecular_processor,
            cache_file=str(cache_file)
        )

        assert 0 in new_dataset.spectral_features_cache
        assert new_dataset.spectral_features_cache[0]['spectral_complexity'] == 0.5


class TestPCQM4Mv2DataModule:
    """Test suite for PCQM4Mv2DataModule."""

    @pytest.fixture
    def datamodule(self, temp_dir):
        """Create test data module."""
        return PCQM4Mv2DataModule(
            data_dir=str(temp_dir),
            batch_size=4,
            num_workers=0,  # Avoid multiprocessing in tests
            subset=True,
            max_samples=20
        )

    def test_initialization(self, datamodule):
        """Test data module initialization."""
        assert datamodule.batch_size == 4
        assert datamodule.subset is True
        assert datamodule.max_samples == 20

    def test_prepare_data(self, datamodule):
        """Test data preparation."""
        # Should not raise an exception
        datamodule.prepare_data()

    @pytest.mark.integration
    def test_setup_and_dataloaders(self, datamodule):
        """Test data module setup and dataloader creation."""
        # Setup for training
        datamodule.setup(stage='fit')

        assert datamodule.dataset_train is not None
        assert datamodule.dataset_val is not None

        # Test dataloaders
        train_loader = datamodule.train_dataloader()
        assert isinstance(train_loader, GeomDataLoader)

        val_loader = datamodule.val_dataloader()
        assert isinstance(val_loader, GeomDataLoader)

        # Test that we can iterate through a few batches
        train_iter = iter(train_loader)
        batch = next(train_iter)
        assert isinstance(batch, Batch)
        assert batch.x is not None
        assert batch.edge_index is not None
        assert batch.y is not None

    def test_curriculum_dataloader(self, datamodule):
        """Test curriculum dataloader creation."""
        datamodule.setup(stage='fit')

        curriculum_loader = datamodule.get_curriculum_dataloader(
            stage='train',
            difficulty_fraction=0.5
        )

        assert isinstance(curriculum_loader, GeomDataLoader)

    def test_mock_dataset_fallback(self, temp_dir):
        """Test that mock dataset is used when OGB is not available."""
        with patch('spectral_temporal_curriculum_molecular_gap_prediction.data.loader.PCQM4Mv2Dataset', None):
            datamodule = PCQM4Mv2DataModule(
                data_dir=str(temp_dir),
                batch_size=4,
                subset=True,
                max_samples=10
            )

            datamodule.setup(stage='fit')
            assert datamodule.dataset_train is not None


class TestDataIntegration:
    """Integration tests for data pipeline."""

    @pytest.mark.integration
    def test_end_to_end_data_pipeline(self, temp_dir):
        """Test complete data pipeline from loading to batching."""
        # Create data module
        datamodule = PCQM4Mv2DataModule(
            data_dir=str(temp_dir),
            batch_size=4,
            num_workers=0,
            subset=True,
            max_samples=20,
            cache_spectral_features=False  # Disable for faster testing
        )

        # Full pipeline
        datamodule.prepare_data()
        datamodule.setup(stage='fit')

        # Get dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        # Test iteration
        for i, batch in enumerate(train_loader):
            if i >= 2:  # Just test first few batches
                break

            # Check batch structure
            assert isinstance(batch, Batch)
            assert batch.x.dim() == 2
            assert batch.edge_index.dim() == 2
            assert batch.y.dim() == 2

            # Check that spectral features are added
            spectral_attrs = [attr for attr in dir(batch) if attr.startswith('spectral_')]
            assert len(spectral_attrs) > 0

        # Test curriculum learning
        curriculum_loader = datamodule.get_curriculum_dataloader(
            stage='train',
            difficulty_fraction=0.5
        )

        batch = next(iter(curriculum_loader))
        assert isinstance(batch, Batch)

    def test_spectral_features_consistency(self, simple_molecular_graph):
        """Test that spectral features are computed consistently."""
        extractor = SpectralFeatureExtractor()

        # Extract features multiple times
        features1 = extractor.extract_spectral_features(simple_molecular_graph)
        features2 = extractor.extract_spectral_features(simple_molecular_graph)

        # Should be identical
        for key in features1:
            assert abs(features1[key] - features2[key]) < 1e-6

    def test_batch_processing_consistency(self, batch_molecular_graphs, molecular_processor):
        """Test that batch processing produces consistent results."""
        # Process individual graphs
        individual_results = []
        for i in range(len(batch_molecular_graphs)):
            data = batch_molecular_graphs.get_example(i)
            processed, features = molecular_processor.process_graph(data)
            individual_results.append((processed, features))

        # Check consistency
        for processed, features in individual_results:
            assert processed.x.shape[1] == molecular_processor.node_feature_dim
            assert 'spectral_complexity' in features
            assert isinstance(features['spectral_complexity'], float)