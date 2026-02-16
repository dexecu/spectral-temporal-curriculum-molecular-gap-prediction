"""Comprehensive tests for model modules with thorough coverage."""

import unittest
import sys
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Disable debug logging for cleaner test output
logging.getLogger().setLevel(logging.WARNING)

try:
    from spectral_temporal_curriculum_molecular_gap_prediction.models.model import (
        ChebyshevSpectralConv, SpectralFilterBank, MessagePassingEncoder,
        DualViewFusionModule, SpectralTemporalNet
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestChebyshevSpectralConv(unittest.TestCase):
    """Test cases for ChebyshevSpectralConv layer."""

    def setUp(self):
        """Set up test fixtures."""
        if not MODELS_AVAILABLE:
            self.skipTest(f"Model imports failed: {IMPORT_ERROR}")

        self.in_channels = 64
        self.out_channels = 128
        self.K = 5

    def test_initialization_valid_params(self):
        """Test layer initialization with valid parameters."""
        layer = ChebyshevSpectralConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K
        )

        self.assertEqual(layer.in_channels, self.in_channels)
        self.assertEqual(layer.out_channels, self.out_channels)
        self.assertEqual(layer.K, self.K)
        self.assertEqual(layer.normalization, 'sym')
        self.assertIsNotNone(layer.weight)
        self.assertIsNotNone(layer.bias)

    def test_initialization_invalid_params(self):
        """Test layer initialization with invalid parameters."""
        with self.assertRaises(TypeError):
            ChebyshevSpectralConv(in_channels=0, out_channels=self.out_channels)

        with self.assertRaises(TypeError):
            ChebyshevSpectralConv(in_channels=self.in_channels, out_channels=-1)

        with self.assertRaises(ValueError):
            ChebyshevSpectralConv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                K=0
            )

        with self.assertRaises(ValueError):
            ChebyshevSpectralConv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                normalization='invalid'
            )

    def test_forward_pass_valid_input(self):
        """Test forward pass with valid input."""
        layer = ChebyshevSpectralConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K
        )

        # Create test data
        num_nodes = 10
        x = torch.randn(num_nodes, self.in_channels)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        # Forward pass
        output = layer(x, edge_index)

        self.assertEqual(output.shape, (num_nodes, self.out_channels))
        self.assertFalse(torch.isnan(output).any())

    def test_forward_pass_invalid_input(self):
        """Test forward pass with invalid input."""
        layer = ChebyshevSpectralConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K
        )

        # Wrong input dimensions
        with self.assertRaises(RuntimeError):
            x = torch.randn(10, self.in_channels + 1)  # Wrong feature dimension
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            layer(x, edge_index)

        # Invalid edge_index shape
        with self.assertRaises(ValueError):
            x = torch.randn(10, self.in_channels)
            edge_index = torch.tensor([0, 1, 2], dtype=torch.long)  # Wrong shape
            layer(x, edge_index)

        # Invalid node indices
        with self.assertRaises(ValueError):
            x = torch.randn(5, self.in_channels)
            edge_index = torch.tensor([[0, 10], [1, 2]], dtype=torch.long)  # Index 10 > 4
            layer(x, edge_index)

    def test_parameter_initialization(self):
        """Test proper parameter initialization."""
        layer = ChebyshevSpectralConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K
        )

        # Check weight tensor shape
        self.assertEqual(layer.weight.shape, (self.K, self.in_channels, self.out_channels))

        # Check bias tensor shape
        self.assertEqual(layer.bias.shape, (self.out_channels,))

        # Check parameters are properly initialized (not zero)
        self.assertFalse(torch.allclose(layer.weight, torch.zeros_like(layer.weight)))


class TestSpectralFilterBank(unittest.TestCase):
    """Test cases for SpectralFilterBank module."""

    def setUp(self):
        """Set up test fixtures."""
        if not MODELS_AVAILABLE:
            self.skipTest(f"Model imports failed: {IMPORT_ERROR}")

        self.in_channels = 64
        self.out_channels = 32
        self.num_filters = 4

    def test_initialization_valid_params(self):
        """Test filter bank initialization with valid parameters."""
        bank = SpectralFilterBank(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_filters=self.num_filters
        )

        self.assertEqual(bank.in_channels, self.in_channels)
        self.assertEqual(bank.out_channels, self.out_channels)
        self.assertEqual(bank.num_filters, self.num_filters)
        self.assertEqual(len(bank.filters), self.num_filters)

    def test_initialization_invalid_params(self):
        """Test filter bank initialization with invalid parameters."""
        with self.assertRaises(TypeError):
            SpectralFilterBank(in_channels=0, out_channels=self.out_channels)

        with self.assertRaises(ValueError):
            SpectralFilterBank(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                num_filters=0
            )

        with self.assertRaises(ValueError):
            SpectralFilterBank(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                dropout=1.5  # Invalid dropout
            )

    def test_forward_pass_valid_input(self):
        """Test forward pass with valid input."""
        bank = SpectralFilterBank(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_filters=self.num_filters
        )

        # Create test data
        num_nodes = 15
        x = torch.randn(num_nodes, self.in_channels)
        edge_index = torch.tensor([[0, 1, 2, 1], [1, 2, 0, 0]], dtype=torch.long)

        # Forward pass
        output = bank(x, edge_index)

        expected_out_dim = self.num_filters * self.out_channels
        self.assertEqual(output.shape, (num_nodes, expected_out_dim))
        self.assertFalse(torch.isnan(output).any())

    def test_forward_pass_invalid_input(self):
        """Test forward pass with invalid input."""
        bank = SpectralFilterBank(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_filters=self.num_filters
        )

        # Wrong input dimensions
        with self.assertRaises(RuntimeError):
            x = torch.randn(10, self.in_channels + 5)  # Wrong feature dimension
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            bank(x, edge_index)


class TestSpectralTemporalNet(unittest.TestCase):
    """Test cases for the main SpectralTemporalNet model."""

    def setUp(self):
        """Set up test fixtures."""
        if not MODELS_AVAILABLE:
            self.skipTest(f"Model imports failed: {IMPORT_ERROR}")

        self.node_features = 64
        self.edge_features = 32
        self.hidden_dim = 128

    def test_initialization_valid_params(self):
        """Test model initialization with valid parameters."""
        model = SpectralTemporalNet(
            node_features=self.node_features,
            edge_features=self.edge_features,
            hidden_dim=self.hidden_dim
        )

        self.assertEqual(model.hidden_dim, self.hidden_dim)
        self.assertEqual(model.output_dim, 1)
        self.assertIsNotNone(model.node_proj)
        self.assertIsNotNone(model.edge_proj)

    def test_initialization_invalid_params(self):
        """Test model initialization with invalid parameters."""
        with self.assertRaises(TypeError):
            SpectralTemporalNet(node_features=0)

        with self.assertRaises(ValueError):
            SpectralTemporalNet(mp_layers=0)

        with self.assertRaises(ValueError):
            SpectralTemporalNet(fusion_type='invalid')

        with self.assertRaises(ValueError):
            SpectralTemporalNet(pooling='invalid')

    def test_forward_pass_single_graph(self):
        """Test forward pass with a single molecular graph."""
        model = SpectralTemporalNet(
            node_features=self.node_features,
            edge_features=self.edge_features,
            hidden_dim=self.hidden_dim,
            output_dim=1
        )
        model.eval()  # Set to evaluation mode

        # Create test molecular graph
        num_nodes = 20
        num_edges = 30

        x = torch.randn(num_nodes, self.node_features)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, self.edge_features)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Forward pass
        with torch.no_grad():
            output = model(data)

        self.assertEqual(output.shape, (1, 1))  # Single graph, single output
        self.assertFalse(torch.isnan(output).any())

    def test_forward_pass_batch(self):
        """Test forward pass with batched molecular graphs."""
        model = SpectralTemporalNet(
            node_features=self.node_features,
            edge_features=self.edge_features,
            hidden_dim=self.hidden_dim,
            output_dim=1
        )
        model.eval()

        # Create batch of test graphs
        graphs = []
        for _ in range(3):
            num_nodes = torch.randint(5, 15, (1,)).item()
            num_edges = torch.randint(num_nodes, num_nodes * 2, (1,)).item()

            x = torch.randn(num_nodes, self.node_features)
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            edge_attr = torch.randn(num_edges, self.edge_features)

            graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

        batch = Batch.from_data_list(graphs)

        # Forward pass
        with torch.no_grad():
            output = model(batch)

        self.assertEqual(output.shape, (3, 1))  # 3 graphs, single output each
        self.assertFalse(torch.isnan(output).any())

    def test_forward_pass_invalid_input(self):
        """Test forward pass with invalid input."""
        model = SpectralTemporalNet(
            node_features=self.node_features,
            edge_features=self.edge_features,
            hidden_dim=self.hidden_dim
        )

        # Missing required attributes
        with self.assertRaises(AttributeError):
            data = Data(edge_index=torch.tensor([[0, 1], [1, 0]]))  # No x
            model(data)

        with self.assertRaises(AttributeError):
            data = Data(x=torch.randn(5, self.node_features))  # No edge_index
            model(data)

    def test_attention_weights_extraction(self):
        """Test attention weights extraction for interpretability."""
        model = SpectralTemporalNet(
            node_features=self.node_features,
            edge_features=self.edge_features,
            hidden_dim=self.hidden_dim,
            pooling='attention'
        )
        model.eval()

        # Create test data
        num_nodes = 10
        x = torch.randn(num_nodes, self.node_features)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_attr = torch.randn(3, self.edge_features)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Extract attention weights
        with torch.no_grad():
            attention_weights = model.get_attention_weights(data)

        self.assertIsInstance(attention_weights, dict)
        # Check that pooling attention is included for attention pooling
        if model.pooling == 'attention':
            self.assertIn('pooling', attention_weights)

    def test_parameter_count(self):
        """Test that model has reasonable parameter count."""
        model = SpectralTemporalNet(
            node_features=self.node_features,
            edge_features=self.edge_features,
            hidden_dim=self.hidden_dim
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Should have reasonable number of parameters (not too few, not too many)
        self.assertGreater(total_params, 1000)  # At least 1K parameters
        self.assertLess(total_params, 50_000_000)  # Less than 50M parameters
        self.assertEqual(total_params, trainable_params)  # All should be trainable

    def test_different_pooling_methods(self):
        """Test model with different pooling methods."""
        pooling_methods = ['mean', 'max', 'sum', 'attention']

        for pooling in pooling_methods:
            with self.subTest(pooling=pooling):
                model = SpectralTemporalNet(
                    node_features=self.node_features,
                    edge_features=self.edge_features,
                    hidden_dim=self.hidden_dim,
                    pooling=pooling
                )

                # Create test data
                num_nodes = 10
                x = torch.randn(num_nodes, self.node_features)
                edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
                edge_attr = torch.randn(3, self.edge_features)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

                # Forward pass should work for all pooling methods
                with torch.no_grad():
                    output = model(data)

                self.assertEqual(output.shape, (1, 1))
                self.assertFalse(torch.isnan(output).any())


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model components."""

    def setUp(self):
        """Set up test fixtures."""
        if not MODELS_AVAILABLE:
            self.skipTest(f"Model imports failed: {IMPORT_ERROR}")

    def test_gradient_flow(self):
        """Test that gradients flow through the model properly."""
        model = SpectralTemporalNet(
            node_features=32,
            edge_features=16,
            hidden_dim=64,
            output_dim=1
        )

        # Create test data
        num_nodes = 8
        x = torch.randn(num_nodes, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_attr = torch.randn(4, 16)
        targets = torch.randn(1, 1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Forward and backward pass
        output = model(data)
        loss = nn.MSELoss()(output, targets)
        loss.backward()

        # Check that gradients exist and are reasonable
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")

        # Should have gradients flowing through most parameters
        self.assertGreater(len(grad_norms), 0)
        self.assertGreater(sum(grad_norms), 0)  # Some gradients should be non-zero

    def test_model_device_compatibility(self):
        """Test model works on CPU (and GPU if available)."""
        model = SpectralTemporalNet(
            node_features=16,
            edge_features=8,
            hidden_dim=32
        )

        # Test on CPU
        device = torch.device('cpu')
        model = model.to(device)

        num_nodes = 5
        x = torch.randn(num_nodes, 16, device=device)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long, device=device)
        edge_attr = torch.randn(3, 8, device=device)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        with torch.no_grad():
            output = model(data)

        self.assertEqual(output.device, device)
        self.assertFalse(torch.isnan(output).any())

        # Test on GPU if available
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            model = model.to(device)
            data = data.to(device)

            with torch.no_grad():
                output = model(data)

            self.assertEqual(output.device, device)
            self.assertFalse(torch.isnan(output).any())


if __name__ == '__main__':
    unittest.main(verbosity=2)