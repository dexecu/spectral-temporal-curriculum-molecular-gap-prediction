"""Tests for model architecture modules."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch

from spectral_temporal_curriculum_molecular_gap_prediction.models.model import (
    ChebyshevSpectralConv, SpectralFilterBank, MessagePassingEncoder,
    DualViewFusionModule, SpectralTemporalNet
)


class TestChebyshevSpectralConv:
    """Test suite for ChebyshevSpectralConv layer."""

    def test_initialization(self):
        """Test Chebyshev convolution initialization."""
        conv = ChebyshevSpectralConv(
            in_channels=16,
            out_channels=32,
            K=5
        )

        assert conv.in_channels == 16
        assert conv.out_channels == 32
        assert conv.K == 5
        assert conv.weight.shape == (5, 16, 32)

    def test_forward_pass(self, simple_molecular_graph):
        """Test forward pass through Chebyshev convolution."""
        conv = ChebyshevSpectralConv(
            in_channels=9,  # Match input features
            out_channels=16,
            K=3
        )

        # Forward pass
        output = conv(
            simple_molecular_graph.x,
            simple_molecular_graph.edge_index
        )

        assert output.shape == (simple_molecular_graph.num_nodes, 16)
        assert not torch.isnan(output).any()

    def test_different_k_values(self, simple_molecular_graph):
        """Test convolution with different K values."""
        for k in [1, 3, 5, 10]:
            conv = ChebyshevSpectralConv(
                in_channels=9,
                out_channels=16,
                K=k
            )

            output = conv(
                simple_molecular_graph.x,
                simple_molecular_graph.edge_index
            )

            assert output.shape == (simple_molecular_graph.num_nodes, 16)
            assert not torch.isnan(output).any()

    def test_parameter_initialization(self):
        """Test that parameters are properly initialized."""
        conv = ChebyshevSpectralConv(in_channels=10, out_channels=20, K=5)

        # Check weight initialization
        assert conv.weight.requires_grad
        assert not torch.isnan(conv.weight).any()

        # Check bias initialization
        if conv.bias is not None:
            assert conv.bias.requires_grad
            assert not torch.isnan(conv.bias).any()

    def test_no_bias(self):
        """Test convolution without bias."""
        conv = ChebyshevSpectralConv(
            in_channels=10,
            out_channels=20,
            K=3,
            bias=False
        )

        assert conv.bias is None


class TestSpectralFilterBank:
    """Test suite for SpectralFilterBank module."""

    def test_initialization(self):
        """Test SpectralFilterBank initialization."""
        filter_bank = SpectralFilterBank(
            in_channels=16,
            out_channels=8,
            num_filters=4,
            max_chebyshev_order=10
        )

        assert filter_bank.in_channels == 16
        assert filter_bank.out_channels == 8
        assert filter_bank.num_filters == 4
        assert len(filter_bank.filters) == 4

    def test_forward_pass(self, simple_molecular_graph):
        """Test forward pass through filter bank."""
        filter_bank = SpectralFilterBank(
            in_channels=9,
            out_channels=8,
            num_filters=3,
            max_chebyshev_order=6
        )

        output = filter_bank(
            simple_molecular_graph.x,
            simple_molecular_graph.edge_index
        )

        expected_output_dim = 3 * 8  # num_filters * out_channels
        assert output.shape == (simple_molecular_graph.num_nodes, expected_output_dim)
        assert not torch.isnan(output).any()

    def test_different_filter_configurations(self, simple_molecular_graph):
        """Test different filter bank configurations."""
        configs = [
            (2, 4, 5),   # (num_filters, out_channels, max_order)
            (4, 8, 10),
            (6, 12, 15)
        ]

        for num_filters, out_channels, max_order in configs:
            filter_bank = SpectralFilterBank(
                in_channels=9,
                out_channels=out_channels,
                num_filters=num_filters,
                max_chebyshev_order=max_order
            )

            output = filter_bank(
                simple_molecular_graph.x,
                simple_molecular_graph.edge_index
            )

            expected_dim = num_filters * out_channels
            assert output.shape == (simple_molecular_graph.num_nodes, expected_dim)


class TestMessagePassingEncoder:
    """Test suite for MessagePassingEncoder module."""

    def test_initialization(self):
        """Test MessagePassingEncoder initialization."""
        encoder = MessagePassingEncoder(
            in_channels=16,
            hidden_channels=32,
            out_channels=64,
            num_layers=3,
            gnn_type='gin'
        )

        assert encoder.num_layers == 3
        assert encoder.gnn_type == 'gin'
        assert len(encoder.convs) == 3
        assert len(encoder.batch_norms) == 3

    def test_forward_pass(self, simple_molecular_graph):
        """Test forward pass through message passing encoder."""
        encoder = MessagePassingEncoder(
            in_channels=9,
            hidden_channels=16,
            out_channels=32,
            num_layers=2,
            gnn_type='gin'
        )

        output = encoder(
            simple_molecular_graph.x,
            simple_molecular_graph.edge_index
        )

        assert output.shape == (simple_molecular_graph.num_nodes, 32)
        assert not torch.isnan(output).any()

    def test_different_gnn_types(self, simple_molecular_graph):
        """Test different GNN architectures."""
        gnn_types = ['gin', 'gcn', 'gat']

        for gnn_type in gnn_types:
            encoder = MessagePassingEncoder(
                in_channels=9,
                hidden_channels=16,
                out_channels=32,
                num_layers=2,
                gnn_type=gnn_type
            )

            output = encoder(
                simple_molecular_graph.x,
                simple_molecular_graph.edge_index
            )

            assert output.shape == (simple_molecular_graph.num_nodes, 32)
            assert not torch.isnan(output).any()

    def test_different_layer_numbers(self, simple_molecular_graph):
        """Test encoders with different numbers of layers."""
        for num_layers in [1, 2, 3, 5]:
            encoder = MessagePassingEncoder(
                in_channels=9,
                hidden_channels=16,
                out_channels=32,
                num_layers=num_layers,
                gnn_type='gin'
            )

            output = encoder(
                simple_molecular_graph.x,
                simple_molecular_graph.edge_index
            )

            assert output.shape == (simple_molecular_graph.num_nodes, 32)
            assert len(encoder.convs) == num_layers

    def test_activation_functions(self, simple_molecular_graph):
        """Test different activation functions."""
        activations = ['relu', 'gelu', 'elu']

        for activation in activations:
            encoder = MessagePassingEncoder(
                in_channels=9,
                hidden_channels=16,
                out_channels=32,
                num_layers=2,
                activation=activation
            )

            output = encoder(
                simple_molecular_graph.x,
                simple_molecular_graph.edge_index
            )

            assert output.shape == (simple_molecular_graph.num_nodes, 32)


class TestDualViewFusionModule:
    """Test suite for DualViewFusionModule."""

    def test_initialization(self):
        """Test DualViewFusionModule initialization."""
        fusion = DualViewFusionModule(
            mp_channels=32,
            spectral_channels=24,
            fusion_channels=48,
            fusion_type='attention'
        )

        assert fusion.mp_channels == 32
        assert fusion.spectral_channels == 24
        assert fusion.fusion_channels == 48
        assert fusion.fusion_type == 'attention'

    def test_concat_fusion(self):
        """Test concatenation fusion strategy."""
        fusion = DualViewFusionModule(
            mp_channels=16,
            spectral_channels=12,
            fusion_channels=20,
            fusion_type='concat'
        )

        mp_features = torch.randn(10, 16)
        spectral_features = torch.randn(10, 12)

        output = fusion(mp_features, spectral_features)

        assert output.shape == (10, 20)
        assert not torch.isnan(output).any()

    def test_attention_fusion(self):
        """Test attention fusion strategy."""
        fusion = DualViewFusionModule(
            mp_channels=16,
            spectral_channels=12,
            fusion_channels=20,
            fusion_type='attention'
        )

        mp_features = torch.randn(10, 16)
        spectral_features = torch.randn(10, 12)

        output = fusion(mp_features, spectral_features)

        assert output.shape == (10, 20)
        assert not torch.isnan(output).any()

    def test_cross_attention_fusion(self):
        """Test cross-attention fusion strategy."""
        fusion = DualViewFusionModule(
            mp_channels=16,
            spectral_channels=12,
            fusion_channels=20,
            fusion_type='cross_attention'
        )

        mp_features = torch.randn(10, 16)
        spectral_features = torch.randn(10, 12)

        output = fusion(mp_features, spectral_features)

        assert output.shape == (10, 20)
        assert not torch.isnan(output).any()

    def test_invalid_fusion_type(self):
        """Test that invalid fusion type raises error."""
        with pytest.raises(ValueError):
            DualViewFusionModule(
                mp_channels=16,
                spectral_channels=12,
                fusion_channels=20,
                fusion_type='invalid'
            )


class TestSpectralTemporalNet:
    """Test suite for the complete SpectralTemporalNet model."""

    def test_initialization(self):
        """Test SpectralTemporalNet initialization."""
        model = SpectralTemporalNet(
            node_features=16,
            edge_features=8,
            hidden_dim=32,
            mp_layers=2,
            num_spectral_filters=3,
            max_chebyshev_order=10
        )

        assert model.hidden_dim == 32
        assert isinstance(model.mp_encoder, MessagePassingEncoder)
        assert isinstance(model.spectral_encoder, SpectralFilterBank)
        assert isinstance(model.fusion, DualViewFusionModule)

    def test_forward_pass_single_graph(self, simple_molecular_graph):
        """Test forward pass with single graph."""
        model = SpectralTemporalNet(
            node_features=9,
            edge_features=3,
            hidden_dim=32,
            mp_layers=2,
            num_spectral_filters=4
        )

        # Convert to batch format
        batch = Batch.from_data_list([simple_molecular_graph])

        output = model(batch)

        assert output.shape == (1, 1)  # batch_size=1, output_dim=1
        assert not torch.isnan(output).any()

    def test_forward_pass_batch(self, batch_molecular_graphs):
        """Test forward pass with batch of graphs."""
        model = SpectralTemporalNet(
            node_features=9,
            edge_features=3,
            hidden_dim=32,
            mp_layers=2,
            num_spectral_filters=4
        )

        output = model(batch_molecular_graphs)

        batch_size = batch_molecular_graphs.num_graphs
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()

    def test_different_pooling_methods(self, batch_molecular_graphs):
        """Test different graph pooling methods."""
        pooling_methods = ['mean', 'max', 'sum', 'attention']

        for pooling in pooling_methods:
            model = SpectralTemporalNet(
                node_features=9,
                edge_features=3,
                hidden_dim=32,
                pooling=pooling
            )

            output = model(batch_molecular_graphs)

            batch_size = batch_molecular_graphs.num_graphs
            assert output.shape == (batch_size, 1)
            assert not torch.isnan(output).any()

    def test_different_fusion_types(self, batch_molecular_graphs):
        """Test different dual-view fusion strategies."""
        fusion_types = ['concat', 'attention', 'cross_attention']

        for fusion_type in fusion_types:
            model = SpectralTemporalNet(
                node_features=9,
                edge_features=3,
                hidden_dim=32,
                fusion_type=fusion_type
            )

            output = model(batch_molecular_graphs)

            batch_size = batch_molecular_graphs.num_graphs
            assert output.shape == (batch_size, 1)
            assert not torch.isnan(output).any()

    def test_parameter_count(self):
        """Test parameter count calculation."""
        model = SpectralTemporalNet(
            node_features=16,
            edge_features=8,
            hidden_dim=64,
            mp_layers=3,
            num_spectral_filters=4
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable

    def test_gradient_flow(self, batch_molecular_graphs):
        """Test that gradients flow through the model."""
        model = SpectralTemporalNet(
            node_features=9,
            edge_features=3,
            hidden_dim=32
        )

        # Forward pass
        output = model(batch_molecular_graphs)

        # Dummy loss
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_model_evaluation_mode(self, batch_molecular_graphs):
        """Test model in evaluation mode."""
        model = SpectralTemporalNet(
            node_features=9,
            edge_features=3,
            hidden_dim=32
        )

        model.eval()

        with torch.no_grad():
            output = model(batch_molecular_graphs)

        assert not torch.isnan(output).any()
        assert output.requires_grad is False

    def test_attention_weights_extraction(self, batch_molecular_graphs):
        """Test attention weights extraction for interpretability."""
        model = SpectralTemporalNet(
            node_features=9,
            edge_features=3,
            hidden_dim=32,
            pooling='attention'
        )

        attention_weights = model.get_attention_weights(batch_molecular_graphs)

        assert isinstance(attention_weights, dict)

    @pytest.mark.gpu
    def test_gpu_compatibility(self, batch_molecular_graphs, device):
        """Test model GPU compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = SpectralTemporalNet(
            node_features=9,
            edge_features=3,
            hidden_dim=32
        ).to(device)

        batch_molecular_graphs = batch_molecular_graphs.to(device)

        output = model(batch_molecular_graphs)

        assert output.device == device
        assert not torch.isnan(output).any()

    def test_different_output_dimensions(self, batch_molecular_graphs):
        """Test model with different output dimensions."""
        for output_dim in [1, 2, 5, 10]:
            model = SpectralTemporalNet(
                node_features=9,
                edge_features=3,
                hidden_dim=32,
                output_dim=output_dim
            )

            output = model(batch_molecular_graphs)

            batch_size = batch_molecular_graphs.num_graphs
            assert output.shape == (batch_size, output_dim)

    def test_model_reproducibility(self, batch_molecular_graphs):
        """Test model output reproducibility."""
        torch.manual_seed(42)

        model = SpectralTemporalNet(
            node_features=9,
            edge_features=3,
            hidden_dim=32
        )

        output1 = model(batch_molecular_graphs)

        # Reset seed and run again
        torch.manual_seed(42)

        model2 = SpectralTemporalNet(
            node_features=9,
            edge_features=3,
            hidden_dim=32
        )

        output2 = model2(batch_molecular_graphs)

        # Should be close (not exactly equal due to floating point)
        assert torch.allclose(output1, output2, atol=1e-6)


class TestModelIntegration:
    """Integration tests for the complete model pipeline."""

    @pytest.mark.integration
    def test_end_to_end_model_pipeline(self, batch_molecular_graphs):
        """Test complete model pipeline from input to output."""
        model = SpectralTemporalNet(
            node_features=9,
            edge_features=3,
            hidden_dim=64,
            mp_layers=3,
            num_spectral_filters=4,
            max_chebyshev_order=10,
            fusion_type='cross_attention',
            pooling='attention'
        )

        # Test training mode
        model.train()
        train_output = model(batch_molecular_graphs)

        # Test evaluation mode
        model.eval()
        with torch.no_grad():
            eval_output = model(batch_molecular_graphs)

        batch_size = batch_molecular_graphs.num_graphs
        assert train_output.shape == (batch_size, 1)
        assert eval_output.shape == (batch_size, 1)

        # Outputs should be different due to dropout
        assert not torch.allclose(train_output, eval_output, atol=1e-3)

    def test_model_optimization_step(self, batch_molecular_graphs):
        """Test a complete optimization step."""
        model = SpectralTemporalNet(
            node_features=9,
            edge_features=3,
            hidden_dim=32
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Create dummy targets
        targets = torch.randn(batch_molecular_graphs.num_graphs, 1)

        # Forward pass
        predictions = model(batch_molecular_graphs)
        loss = criterion(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that loss is computed
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_model_state_dict_save_load(self, temp_dir):
        """Test model state dict saving and loading."""
        model1 = SpectralTemporalNet(
            node_features=16,
            edge_features=8,
            hidden_dim=32
        )

        # Save state dict
        save_path = temp_dir / "model_state.pt"
        torch.save(model1.state_dict(), save_path)

        # Create new model and load state dict
        model2 = SpectralTemporalNet(
            node_features=16,
            edge_features=8,
            hidden_dim=32
        )

        model2.load_state_dict(torch.load(save_path))

        # Check that parameters are the same
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)