"""Spectral-Temporal Graph Neural Network for Molecular Property Prediction.

This module implements the core dual-view architecture that combines message-passing
GNNs with spectral graph wavelets via learnable Chebyshev filter banks.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, MessagePassing, global_mean_pool,
    global_max_pool, global_add_pool
)
from torch_geometric.utils import get_laplacian, degree
from torch_geometric.typing import Adj, OptTensor

logger = logging.getLogger(__name__)


class ChebyshevSpectralConv(MessagePassing):
    """Learnable Chebyshev spectral convolution layer.

    Implements spectral graph convolution using Chebyshev polynomial approximation
    of the graph Laplacian eigendecomposition. This layer captures multi-scale
    spectral patterns in molecular graphs through learnable filter coefficients.

    The layer computes T_k(L̃)x where T_k are Chebyshev polynomials of order k,
    L̃ is the normalized and scaled Laplacian matrix, and x are node features.

    Attributes:
        in_channels (int): Number of input node features.
        out_channels (int): Number of output node features.
        K (int): Order of Chebyshev polynomial approximation.
        normalization (str): Laplacian normalization type.
        weight (torch.nn.Parameter): Learnable filter coefficients.
        bias (Optional[torch.nn.Parameter]): Optional bias parameters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 10,
        normalization: str = 'sym',
        bias: bool = True,
        **kwargs
    ):
        """Initialize Chebyshev spectral convolution.

        Args:
            in_channels (int): Number of input node features.
            out_channels (int): Number of output node features.
            K (int, optional): Order of Chebyshev polynomial approximation.
                Higher orders capture more complex spectral patterns but increase
                computational cost. Defaults to 10.
            normalization (str, optional): Laplacian normalization type.
                'sym' for symmetric normalization, 'rw' for random walk.
                Defaults to 'sym'.
            bias (bool, optional): Whether to include learnable bias parameters.
                Defaults to True.
            **kwargs: Additional arguments passed to MessagePassing parent class.

        Raises:
            ValueError: If K is less than 1 or normalization is invalid.
            TypeError: If in_channels or out_channels are not positive integers.
        """
        # Input validation
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise TypeError(f"in_channels must be a positive integer, got {in_channels}")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise TypeError(f"out_channels must be a positive integer, got {out_channels}")
        if not isinstance(K, int) or K < 1:
            raise ValueError(f"K must be a positive integer >= 1, got {K}")
        if normalization not in ['sym', 'rw']:
            raise ValueError(f"normalization must be 'sym' or 'rw', got {normalization}")

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization

        # Learnable filter coefficients
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None
    ) -> torch.Tensor:
        """Forward pass with Chebyshev polynomial approximation.

        Computes spectral convolution using Chebyshev polynomials:
        h_out = Σ_{k=0}^{K-1} T_k(L̃) X W_k + b

        Args:
            x (torch.Tensor): Node feature matrix of shape [N, in_channels].
            edge_index (Adj): Graph connectivity in COO format [2, E].
            edge_weight (OptTensor, optional): Edge weights of shape [E].
                If None, all edges have weight 1. Defaults to None.
            batch (OptTensor, optional): Batch assignment vector [N] for batched graphs.
                Defaults to None.
            lambda_max (OptTensor, optional): Maximum eigenvalue for Laplacian scaling.
                If None, estimated automatically. Defaults to None.

        Returns:
            torch.Tensor: Output node features of shape [N, out_channels].

        Raises:
            RuntimeError: If input tensor dimensions are incompatible.
            ValueError: If edge_index contains invalid indices.

        Note:
            The Laplacian is normalized and scaled to [-1, 1] range for numerical
            stability of Chebyshev polynomial evaluation.
        """
        # Input validation
        if x.dim() != 2 or x.size(1) != self.in_channels:
            raise RuntimeError(
                f"Expected input tensor of shape [N, {self.in_channels}], "
                f"got {list(x.shape)}"
            )

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"Expected edge_index of shape [2, E], got {list(edge_index.shape)}"
            )

        # Check for valid node indices
        max_node_idx = edge_index.max().item()
        if max_node_idx >= x.size(0):
            raise ValueError(
                f"edge_index contains invalid node index {max_node_idx}, "
                f"but only {x.size(0)} nodes available"
            )

        try:
            # Compute normalized Laplacian
            edge_index, edge_weight = get_laplacian(
                edge_index, edge_weight, normalization=self.normalization,
                dtype=x.dtype, num_nodes=x.size(0)
            )
        except Exception as e:
            logger.error(f"Failed to compute Laplacian matrix: {e}")
            raise RuntimeError(f"Laplacian computation failed: {e}") from e

        # Estimate lambda_max if not provided
        if lambda_max is None:
            lambda_max = self._estimate_lambda_max(edge_index, edge_weight, x.size(0))

        # Scale Laplacian to [-1, 1] range for Chebyshev polynomials
        edge_weight = (2.0 / lambda_max) * edge_weight - 1.0

        # Compute Chebyshev polynomials
        Tx_0 = x
        Tx_1 = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        # Initialize output with T_0 term
        out = torch.matmul(Tx_0, self.weight[0])

        # Add T_1 term if K > 1
        if self.K > 1:
            out = out + torch.matmul(Tx_1, self.weight[1])

        # Compute higher-order terms using recurrence relation
        for k in range(2, self.K):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, edge_weight=edge_weight, size=None) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def _estimate_lambda_max(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        num_nodes: int
    ) -> float:
        """Estimate maximum eigenvalue of Laplacian matrix.

        Args:
            edge_index (torch.Tensor): Graph connectivity [2, E].
            edge_weight (Optional[torch.Tensor]): Edge weights [E].
            num_nodes (int): Number of nodes in the graph.

        Returns:
            float: Estimated maximum eigenvalue.

        Note:
            Uses a simple heuristic (2.0) that works well for most molecular graphs.
            For more accuracy, power iteration could be implemented.
        """
        # Simple heuristic: 2.0 works well for most graphs
        # For more accuracy, could use power iteration
        return 2.0

    def message(self, x_j: torch.Tensor, edge_weight: OptTensor) -> torch.Tensor:
        """Construct messages from neighboring nodes.

        Args:
            x_j (torch.Tensor): Features of neighboring nodes [E, in_channels].
            edge_weight (OptTensor): Edge weights for message weighting [E].

        Returns:
            torch.Tensor: Weighted messages from neighbors [E, in_channels].
        """
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class SpectralFilterBank(nn.Module):
    """Learnable spectral filter bank using multiple Chebyshev convolutions.

    Implements a bank of spectral filters with different polynomial orders
    to capture multi-scale spectral patterns in molecular graphs. Each filter
    operates at a different scale, enabling the model to learn both local
    and global graph structure representations.

    The filter bank applies multiple Chebyshev spectral convolutions in parallel,
    then combines their outputs using learned attention weights. This approach
    allows the model to adaptively focus on the most informative spectral scales
    for each graph.

    Attributes:
        in_channels (int): Number of input node features.
        out_channels (int): Number of output features per filter.
        num_filters (int): Number of spectral filters in the bank.
        max_chebyshev_order (int): Maximum polynomial order across filters.
        filters (nn.ModuleList): List of Chebyshev spectral convolution layers.
        filter_attention (nn.Sequential): Attention mechanism for filter combination.
        dropout (nn.Dropout): Dropout layer for regularization.
        layer_norm (nn.LayerNorm): Layer normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int = 4,
        max_chebyshev_order: int = 20,
        dropout: float = 0.1
    ):
        """Initialize spectral filter bank.

        Args:
            in_channels (int): Number of input node features.
            out_channels (int): Number of output features per filter.
            num_filters (int, optional): Number of spectral filters in the bank.
                More filters capture finer spectral details. Defaults to 4.
            max_chebyshev_order (int, optional): Maximum Chebyshev polynomial order.
                Higher orders capture more complex spectral patterns. Defaults to 20.
            dropout (float, optional): Dropout probability for regularization.
                Must be in [0, 1). Defaults to 0.1.

        Raises:
            ValueError: If num_filters < 1, max_chebyshev_order < 1, or dropout not in [0, 1).
            TypeError: If in_channels or out_channels are not positive integers.
        """
        # Input validation
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise TypeError(f"in_channels must be a positive integer, got {in_channels}")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise TypeError(f"out_channels must be a positive integer, got {out_channels}")
        if not isinstance(num_filters, int) or num_filters < 1:
            raise ValueError(f"num_filters must be >= 1, got {num_filters}")
        if not isinstance(max_chebyshev_order, int) or max_chebyshev_order < 1:
            raise ValueError(f"max_chebyshev_order must be >= 1, got {max_chebyshev_order}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.max_chebyshev_order = max_chebyshev_order

        # Create filter bank with different polynomial orders
        self.filters = nn.ModuleList()
        orders = [max(1, (i + 1) * max_chebyshev_order // num_filters) for i in range(num_filters)]

        logger.debug(
            f"Creating SpectralFilterBank with {num_filters} filters, "
            f"orders: {orders}, max_order: {max_chebyshev_order}"
        )

        for i, order in enumerate(orders):
            try:
                filter_layer = ChebyshevSpectralConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    K=order
                )
                self.filters.append(filter_layer)
                logger.debug(f"Created filter {i} with order {order}")
            except Exception as e:
                logger.error(f"Failed to create filter {i} with order {order}: {e}")
                raise

        # Attention mechanism for filter combination
        self.filter_attention = nn.Sequential(
            nn.Linear(num_filters * out_channels, num_filters),
            nn.Softmax(dim=-1)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_filters * out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply spectral filter bank.

        Args:
            x (torch.Tensor): Node features of shape [N, in_channels].
            edge_index (torch.Tensor): Graph connectivity in COO format [2, E].
            edge_weight (Optional[torch.Tensor], optional): Edge weights [E].
                Defaults to None.
            batch (Optional[torch.Tensor], optional): Batch assignment [N].
                Defaults to None.

        Returns:
            torch.Tensor: Multi-scale spectral features of shape
                [N, num_filters * out_channels].

        Raises:
            RuntimeError: If input dimensions are incompatible.
            ValueError: If any filter operation fails.
        """
        # Input validation
        if x.dim() != 2 or x.size(1) != self.in_channels:
            raise RuntimeError(
                f"Expected input tensor of shape [N, {self.in_channels}], "
                f"got {list(x.shape)}"
            )

        logger.debug(
            f"Applying {len(self.filters)} spectral filters to input shape {list(x.shape)}"
        )

        # Apply each spectral filter
        filter_outputs = []
        for i, spectral_filter in enumerate(self.filters):
            try:
                filter_out = spectral_filter(x, edge_index, edge_weight, batch)
                filter_outputs.append(filter_out)
                logger.debug(
                    f"Filter {i} output shape: {list(filter_out.shape)}"
                )
            except Exception as e:
                logger.error(f"Filter {i} failed: {e}")
                raise ValueError(f"Spectral filter {i} computation failed: {e}") from e

        # Concatenate filter outputs
        multi_scale_features = torch.cat(filter_outputs, dim=-1)

        # Apply layer normalization and dropout
        multi_scale_features = self.layer_norm(multi_scale_features)
        multi_scale_features = self.dropout(multi_scale_features)

        return multi_scale_features


class MessagePassingEncoder(nn.Module):
    """Message-passing encoder with multiple GNN layers."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        gnn_type: str = 'gin',
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """Initialize message-passing encoder.

        Args:
            in_channels: Number of input node features
            hidden_channels: Hidden dimension
            out_channels: Output dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ('gin', 'gcn', 'gat')
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        self.num_layers = num_layers
        self.gnn_type = gnn_type.lower()
        self.dropout = dropout

        # Select activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            self.activation = F.relu

        # Build GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels

            # Create GNN layer based on type
            if self.gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, out_dim)
                )
                conv = GINConv(mlp)
            elif self.gnn_type == 'gcn':
                conv = GCNConv(in_dim, out_dim)
            elif self.gnn_type == 'gat':
                conv = GATConv(in_dim, out_dim, heads=4, concat=False)
            else:
                raise ValueError(f"Unknown GNN type: {self.gnn_type}")

            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(out_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through message-passing layers.

        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge features [E, edge_dim] (unused for now)
            batch: Batch assignment [N]

        Returns:
            Node embeddings [N, out_channels]
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)

            if i < len(self.convs) - 1:  # No activation on last layer
                x_new = self.activation(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            x = x_new

        return x


class DualViewFusionModule(nn.Module):
    """Fusion module for combining message-passing and spectral representations."""

    def __init__(
        self,
        mp_channels: int,
        spectral_channels: int,
        fusion_channels: int,
        fusion_type: str = 'attention'
    ):
        """Initialize dual-view fusion module.

        Args:
            mp_channels: Dimension of message-passing features
            spectral_channels: Dimension of spectral features
            fusion_channels: Output dimension after fusion
            fusion_type: Fusion strategy ('concat', 'attention', 'cross_attention')
        """
        super().__init__()

        self.mp_channels = mp_channels
        self.spectral_channels = spectral_channels
        self.fusion_channels = fusion_channels
        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            # Simple concatenation with projection
            self.fusion_proj = nn.Linear(mp_channels + spectral_channels, fusion_channels)

        elif fusion_type == 'attention':
            # Self-attention over concatenated features
            total_channels = mp_channels + spectral_channels
            self.attention = nn.MultiheadAttention(
                embed_dim=total_channels,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.fusion_proj = nn.Linear(total_channels, fusion_channels)

        elif fusion_type == 'cross_attention':
            # Cross-attention between MP and spectral features
            self.mp_proj = nn.Linear(mp_channels, fusion_channels)
            self.spectral_proj = nn.Linear(spectral_channels, fusion_channels)

            self.cross_attention = nn.MultiheadAttention(
                embed_dim=fusion_channels,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.layer_norm = nn.LayerNorm(fusion_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        mp_features: torch.Tensor,
        spectral_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fuse message-passing and spectral representations.

        Args:
            mp_features: Message-passing features [N, mp_channels]
            spectral_features: Spectral features [N, spectral_channels]
            batch: Batch assignment [N]

        Returns:
            Fused features [N, fusion_channels]
        """
        if self.fusion_type == 'concat':
            # Simple concatenation and projection
            combined = torch.cat([mp_features, spectral_features], dim=-1)
            fused = self.fusion_proj(combined)

        elif self.fusion_type == 'attention':
            # Self-attention over concatenated features
            combined = torch.cat([mp_features, spectral_features], dim=-1)
            combined = combined.unsqueeze(0)  # Add sequence dimension

            # Apply self-attention
            attended, _ = self.attention(combined, combined, combined)
            attended = attended.squeeze(0)  # Remove sequence dimension

            fused = self.fusion_proj(attended)

        elif self.fusion_type == 'cross_attention':
            # Cross-attention between views
            mp_proj = self.mp_proj(mp_features).unsqueeze(0)
            spectral_proj = self.spectral_proj(spectral_features).unsqueeze(0)

            # Cross-attention: spectral as query, MP as key/value
            cross_attended, _ = self.cross_attention(spectral_proj, mp_proj, mp_proj)
            fused = cross_attended.squeeze(0)

        # Apply layer norm and dropout
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        return fused


class SpectralTemporalNet(nn.Module):
    """Dual-view spectral-temporal neural network for molecular property prediction.

    This model combines two complementary views of molecular graphs:
    1. Message-passing view: Captures local chemical interactions and bonding patterns
    2. Spectral view: Captures global structural properties through eigendecomposition

    The dual-view architecture enables the model to learn both local chemical
    environments and global molecular topology, leading to improved prediction
    accuracy for molecular properties like HOMO-LUMO gaps.

    The model architecture consists of:
    - Node/edge feature projections
    - Message-passing encoder (GIN/GCN/GAT layers)
    - Spectral filter bank with multiple Chebyshev convolutions
    - Dual-view fusion module (concatenation/attention/cross-attention)
    - Graph-level pooling (mean/max/sum/attention)
    - Multi-layer output head

    Attributes:
        hidden_dim (int): Hidden dimension used throughout the network.
        pooling (str): Graph-level pooling method.
        output_dim (int): Final output dimension.
        node_proj (nn.Linear): Node feature projection layer.
        edge_proj (Optional[nn.Linear]): Edge feature projection layer.
        mp_encoder (MessagePassingEncoder): Message-passing encoder.
        spectral_encoder (SpectralFilterBank): Spectral filter bank.
        fusion (DualViewFusionModule): Dual-view fusion module.
        pool_attention (Optional[nn.Sequential]): Attention pooling layers.
        output_head (nn.Sequential): Final prediction layers.
    """

    def __init__(
        self,
        node_features: int = 128,
        edge_features: int = 64,
        hidden_dim: int = 256,
        mp_layers: int = 4,
        num_spectral_filters: int = 6,
        max_chebyshev_order: int = 20,
        fusion_type: str = 'cross_attention',
        dropout: float = 0.1,
        pooling: str = 'attention',
        output_dim: int = 1
    ):
        """Initialize spectral-temporal network.

        Args:
            node_features (int, optional): Input node feature dimension.
                Molecular graphs typically have 9-128 atomic features. Defaults to 128.
            edge_features (int, optional): Input edge feature dimension.
                Bond features like bond type, aromaticity. Defaults to 64.
            hidden_dim (int, optional): Hidden dimension for all modules.
                Larger values increase model capacity but also computational cost.
                Defaults to 256.
            mp_layers (int, optional): Number of message-passing layers.
                More layers capture larger chemical neighborhoods. Defaults to 4.
            num_spectral_filters (int, optional): Number of spectral filters in bank.
                More filters capture finer spectral details. Defaults to 6.
            max_chebyshev_order (int, optional): Maximum Chebyshev polynomial order.
                Higher orders capture more complex spectral patterns. Defaults to 20.
            fusion_type (str, optional): Type of dual-view fusion.
                Options: 'concat', 'attention', 'cross_attention'. Defaults to 'cross_attention'.
            dropout (float, optional): Dropout probability for regularization.
                Must be in [0, 1). Defaults to 0.1.
            pooling (str, optional): Graph-level pooling method.
                Options: 'mean', 'max', 'sum', 'attention'. Defaults to 'attention'.
            output_dim (int, optional): Output dimension.
                1 for regression (HOMO-LUMO gap), >1 for multi-task. Defaults to 1.

        Raises:
            ValueError: If any parameter is outside valid range.
            TypeError: If parameters are not of expected types.
        """
        # Input validation
        if not isinstance(node_features, int) or node_features <= 0:
            raise TypeError(f"node_features must be a positive integer, got {node_features}")
        if not isinstance(edge_features, int) or edge_features < 0:
            raise TypeError(f"edge_features must be a non-negative integer, got {edge_features}")
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise TypeError(f"hidden_dim must be a positive integer, got {hidden_dim}")
        if not isinstance(mp_layers, int) or mp_layers <= 0:
            raise ValueError(f"mp_layers must be positive, got {mp_layers}")
        if not isinstance(num_spectral_filters, int) or num_spectral_filters <= 0:
            raise ValueError(f"num_spectral_filters must be positive, got {num_spectral_filters}")
        if not isinstance(max_chebyshev_order, int) or max_chebyshev_order <= 0:
            raise ValueError(f"max_chebyshev_order must be positive, got {max_chebyshev_order}")
        if fusion_type not in ['concat', 'attention', 'cross_attention']:
            raise ValueError(f"fusion_type must be one of ['concat', 'attention', 'cross_attention'], got {fusion_type}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if pooling not in ['mean', 'max', 'sum', 'attention']:
            raise ValueError(f"pooling must be one of ['mean', 'max', 'sum', 'attention'], got {pooling}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

        super().__init__()

        logger.info(
            f"Initializing SpectralTemporalNet: hidden_dim={hidden_dim}, "
            f"mp_layers={mp_layers}, num_spectral_filters={num_spectral_filters}, "
            f"fusion_type={fusion_type}, pooling={pooling}"
        )

        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.output_dim = output_dim

        # Input projections
        self.node_proj = nn.Linear(node_features, hidden_dim)
        self.edge_proj = nn.Linear(edge_features, hidden_dim) if edge_features > 0 else None

        # Message-passing encoder
        self.mp_encoder = MessagePassingEncoder(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=mp_layers,
            gnn_type='gin',
            dropout=dropout
        )

        # Spectral filter bank
        self.spectral_encoder = SpectralFilterBank(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_spectral_filters,
            num_filters=num_spectral_filters,
            max_chebyshev_order=max_chebyshev_order,
            dropout=dropout
        )

        # Dual-view fusion
        self.fusion = DualViewFusionModule(
            mp_channels=hidden_dim,
            spectral_channels=hidden_dim,  # num_filters * (hidden_dim // num_filters)
            fusion_channels=hidden_dim,
            fusion_type=fusion_type
        )

        # Graph-level pooling
        if pooling == 'attention':
            self.pool_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        """Forward pass through dual-view network.

        Processes molecular graphs through both message-passing and spectral views,
        fuses the representations, applies graph-level pooling, and predicts
        molecular properties.

        Args:
            data (Union[Data, Batch]): PyTorch Geometric data object containing:
                - x: Node features [N, node_features]
                - edge_index: Graph connectivity [2, E]
                - edge_attr: Optional edge features [E, edge_features]
                - batch: Batch assignment for multiple graphs [N]

        Returns:
            torch.Tensor: Predicted molecular properties of shape [batch_size, output_dim].

        Raises:
            AttributeError: If required data attributes are missing.
            RuntimeError: If tensor shapes are incompatible.
            ValueError: If input data is malformed.

        Note:
            For single graphs, batch assignment is created automatically.
        """
        # Validate required attributes
        if not hasattr(data, 'x') or data.x is None:
            raise AttributeError("Input data must have node features 'x'")
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            raise AttributeError("Input data must have edge connectivity 'edge_index'")

        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        edge_attr = getattr(data, 'edge_attr', None)

        # Create batch assignment for single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Log input shapes for debugging
        logger.debug(
            f"Forward pass: x.shape={list(x.shape)}, "
            f"edge_index.shape={list(edge_index.shape)}, "
            f"batch_size={batch.max().item() + 1 if batch.numel() > 0 else 1}"
        )

        # Additional validation
        if x.dim() != 2:
            raise ValueError(f"Node features must be 2D tensor, got shape {list(x.shape)}")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must be [2, E] tensor, got shape {list(edge_index.shape)}")

        num_nodes = x.size(0)
        batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1

        logger.debug(f"Processing {num_nodes} nodes across {batch_size} graphs")

        try:
            # Input projections
            x = self.node_proj(x)
            logger.debug(f"Node projection output shape: {list(x.shape)}")

            if edge_attr is not None and self.edge_proj is not None:
                edge_attr = self.edge_proj(edge_attr)
                logger.debug(f"Edge projection output shape: {list(edge_attr.shape)}")

        except Exception as e:
            logger.error(f"Input projection failed: {e}")
            raise RuntimeError(f"Feature projection failed: {e}") from e

        try:
            # Message-passing view
            mp_features = self.mp_encoder(x, edge_index, edge_attr, batch)
            logger.debug(f"Message-passing features shape: {list(mp_features.shape)}")
        except Exception as e:
            logger.error(f"Message-passing encoder failed: {e}")
            raise RuntimeError(f"Message-passing encoding failed: {e}") from e

        try:
            # Spectral view
            spectral_features = self.spectral_encoder(x, edge_index, batch=batch)
            logger.debug(f"Spectral features shape: {list(spectral_features.shape)}")
        except Exception as e:
            logger.error(f"Spectral encoder failed: {e}")
            raise RuntimeError(f"Spectral encoding failed: {e}") from e

        try:
            # Dual-view fusion
            fused_features = self.fusion(mp_features, spectral_features, batch)
            logger.debug(f"Fused features shape: {list(fused_features.shape)}")
        except Exception as e:
            logger.error(f"Dual-view fusion failed: {e}")
            raise RuntimeError(f"Feature fusion failed: {e}") from e

        try:
            # Graph-level pooling
            graph_features = self._global_pool(fused_features, batch)
            logger.debug(f"Graph features shape: {list(graph_features.shape)}")
        except Exception as e:
            logger.error(f"Graph pooling failed: {e}")
            raise RuntimeError(f"Graph pooling failed: {e}") from e

        try:
            # Output prediction
            output = self.output_head(graph_features)
            logger.debug(f"Final output shape: {list(output.shape)}")
        except Exception as e:
            logger.error(f"Output head failed: {e}")
            raise RuntimeError(f"Output prediction failed: {e}") from e

        return output

    def _global_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Apply graph-level pooling.

        Args:
            x: Node features [N, hidden_dim]
            batch: Batch assignment [N]

        Returns:
            Graph-level features [batch_size, hidden_dim]
        """
        if self.pooling == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling == 'max':
            return global_max_pool(x, batch)
        elif self.pooling == 'sum':
            return global_add_pool(x, batch)
        elif self.pooling == 'attention':
            # Attention-based pooling
            attention_weights = self.pool_attention(x)  # [N, 1]
            attention_weights = F.softmax(attention_weights, dim=0)

            # Weighted sum over nodes in each graph
            weighted_features = x * attention_weights
            return global_add_pool(weighted_features, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def get_attention_weights(self, data: Union[Data, Batch]) -> Dict[str, torch.Tensor]:
        """Get attention weights for interpretability.

        Args:
            data: Input molecular graphs

        Returns:
            Dictionary of attention weights from different modules
        """
        attention_weights = {}

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Project inputs
        x = self.node_proj(x)

        # Get features from both views
        mp_features = self.mp_encoder(x, edge_index, batch=batch)
        spectral_features = self.spectral_encoder(x, edge_index, batch=batch)

        # Get fusion attention (if applicable)
        if hasattr(self.fusion, 'attention') or hasattr(self.fusion, 'cross_attention'):
            # This would require modifying fusion module to return attention weights
            pass

        # Get pooling attention weights
        if self.pooling == 'attention':
            pool_weights = self.pool_attention(
                self.fusion(mp_features, spectral_features, batch)
            )
            attention_weights['pooling'] = F.softmax(pool_weights, dim=0)

        return attention_weights