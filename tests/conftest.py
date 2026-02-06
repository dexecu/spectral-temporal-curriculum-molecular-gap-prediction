"""Test fixtures and configuration for the test suite."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from pathlib import Path
import tempfile
import shutil

# Add src to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spectral_temporal_curriculum_molecular_gap_prediction.utils.config import Config
from spectral_temporal_curriculum_molecular_gap_prediction.data.preprocessing import (
    MolecularGraphProcessor, SpectralFeatureExtractor
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config():
    """Create a test configuration with minimal settings."""
    config = Config()

    # Override for faster testing
    config.model.hidden_dim = 32
    config.model.mp_layers = 2
    config.model.num_spectral_filters = 2
    config.model.max_chebyshev_order = 5
    config.data.batch_size = 8
    config.data.subset = True
    config.data.max_samples = 100
    config.training.max_epochs = 2
    config.curriculum.warmup_epochs = 1
    config.curriculum.total_epochs = 2

    return config


@pytest.fixture
def simple_molecular_graph():
    """Create a simple molecular graph for testing."""
    # Create a simple benzene-like ring structure
    num_nodes = 6

    # Edge connectivity for a ring
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],  # source nodes
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]   # target nodes
    ], dtype=torch.long)

    # Random node features (atomic properties)
    x = torch.randn(num_nodes, 9)

    # Random edge features (bond properties)
    edge_attr = torch.randn(edge_index.shape[1], 3)

    # Random HOMO-LUMO gap target
    y = torch.tensor([np.random.uniform(1.0, 8.0)])

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=num_nodes
    )


@pytest.fixture
def batch_molecular_graphs(simple_molecular_graph):
    """Create a batch of molecular graphs for testing."""
    # Create several variations of the simple graph
    graphs = []
    for i in range(4):
        graph = simple_molecular_graph.clone()
        # Add some variation
        graph.x += 0.1 * torch.randn_like(graph.x)
        graph.y = torch.tensor([np.random.uniform(1.0, 8.0)])
        graphs.append(graph)

    return Batch.from_data_list(graphs)


@pytest.fixture
def molecular_processor():
    """Create a molecular graph processor for testing."""
    return MolecularGraphProcessor(
        node_feature_dim=32,
        edge_feature_dim=16,
        add_self_loops=True,
        normalize_features=False
    )


@pytest.fixture
def spectral_extractor():
    """Create a spectral feature extractor for testing."""
    return SpectralFeatureExtractor(
        k_eigenvalues=5,
        chebyshev_order_max=10,
        spectral_tolerance=1e-2
    )


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    class MockDataset:
        def __init__(self, size=50):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Generate deterministic but varied molecular graphs
            np.random.seed(idx)
            torch.manual_seed(idx)

            num_nodes = np.random.randint(5, 20)
            num_edges = np.random.randint(num_nodes, num_nodes * 3)

            # Random edge indices
            edge_index = torch.randint(0, num_nodes, (2, num_edges))

            # Random features
            x = torch.randn(num_nodes, 9)
            edge_attr = torch.randn(num_edges, 3)

            # Random target
            y = torch.tensor([np.random.uniform(1.0, 10.0)])

            return Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                num_nodes=num_nodes
            )

    return MockDataset()


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def sample_predictions_targets():
    """Create sample predictions and targets for metric testing."""
    torch.manual_seed(42)
    predictions = torch.randn(100, 1) * 2 + 5  # Around 5 eV with variation
    targets = predictions + 0.5 * torch.randn(100, 1)  # Add noise
    return predictions, targets


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "gpu: mark test as requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark slow tests
        if "slow" in item.name or "integration" in item.name:
            item.add_marker(pytest.mark.slow)

        # Mark GPU tests
        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)