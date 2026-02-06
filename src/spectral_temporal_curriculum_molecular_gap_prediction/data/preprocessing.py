"""Molecular graph preprocessing and spectral feature extraction."""

import logging
import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, get_laplacian
from scipy.sparse.linalg import eigsh
from scipy.special import eval_chebyt

logger = logging.getLogger(__name__)


class SpectralFeatureExtractor:
    """Extracts spectral features from molecular graphs for curriculum learning.

    This class computes spectral properties of molecular graphs that serve as
    complexity indicators for curriculum learning. The spectral features include:
    - Eigenvalue spectrum of the graph Laplacian
    - Spectral gap (difference between first two eigenvalues)
    - Algebraic connectivity (second smallest eigenvalue)
    - Optimal Chebyshev polynomial order for spectral approximation

    These features help order molecular graphs from simple to complex for
    effective curriculum learning in HOMO-LUMO gap prediction.

    Attributes:
        k_eigenvalues (int): Number of smallest eigenvalues to compute.
        chebyshev_order_max (int): Maximum Chebyshev polynomial order to test.
        spectral_tolerance (float): Tolerance for spectral approximation quality.
    """

    def __init__(
        self,
        k_eigenvalues: int = 10,
        chebyshev_order_max: int = 20,
        spectral_tolerance: float = 1e-3
    ):
        """Initialize spectral feature extractor.

        Args:
            k_eigenvalues (int, optional): Number of smallest eigenvalues to compute.
                More eigenvalues provide richer spectral information but increase
                computational cost. Should be much less than graph size. Defaults to 10.
            chebyshev_order_max (int, optional): Maximum Chebyshev polynomial order
                to test for approximation quality. Higher orders can capture more
                complex spectral patterns. Defaults to 20.
            spectral_tolerance (float, optional): Tolerance for spectral approximation
                quality. Smaller values require higher-order polynomials.
                Defaults to 1e-3.

        Raises:
            ValueError: If k_eigenvalues < 1, chebyshev_order_max < 1, or
                spectral_tolerance <= 0.
        """
        # Input validation
        if not isinstance(k_eigenvalues, int) or k_eigenvalues < 1:
            raise ValueError(f"k_eigenvalues must be a positive integer, got {k_eigenvalues}")
        if not isinstance(chebyshev_order_max, int) or chebyshev_order_max < 1:
            raise ValueError(f"chebyshev_order_max must be a positive integer, got {chebyshev_order_max}")
        if not isinstance(spectral_tolerance, (int, float)) or spectral_tolerance <= 0:
            raise ValueError(f"spectral_tolerance must be positive, got {spectral_tolerance}")

        self.k_eigenvalues = k_eigenvalues
        self.chebyshev_order_max = chebyshev_order_max
        self.spectral_tolerance = spectral_tolerance

        logger.debug(
            f"Initialized SpectralFeatureExtractor: k_eigenvalues={k_eigenvalues}, "
            f"chebyshev_order_max={chebyshev_order_max}, spectral_tolerance={spectral_tolerance}"
        )

    def extract_spectral_features(self, data: Data) -> Dict[str, float]:
        """Extract spectral complexity features from molecular graph.

        Computes spectral properties of the molecular graph Laplacian that
        indicate structural complexity. These features are used for curriculum
        learning to order training samples from simple to complex.

        Args:
            data (Data): PyTorch Geometric data object containing:
                - x: Node features [N, F]
                - edge_index: Graph connectivity [2, E]
                - num_nodes: Number of nodes

        Returns:
            Dict[str, float]: Dictionary containing spectral features:
                - spectral_gap: Difference between first two eigenvalues
                - algebraic_connectivity: Second smallest eigenvalue
                - chebyshev_order: Estimated optimal Chebyshev polynomial order
                - num_nodes: Number of nodes in the graph
                - num_edges: Number of edges in the graph
                - density: Graph edge density
                - spectral_complexity: Overall complexity score [0, 1]

        Raises:
            ValueError: If data is invalid or malformed.
            RuntimeError: If spectral computation fails.

        Note:
            Returns default features for empty graphs or when computation fails.
        """
        # Input validation
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            logger.warning("Data missing edge_index, returning default features")
            return self._default_features()

        if not hasattr(data, 'num_nodes') or data.num_nodes <= 0:
            logger.warning("Data has invalid num_nodes, returning default features")
            return self._default_features()

        logger.debug(f"Extracting spectral features from graph with {data.num_nodes} nodes")

        try:
            # Convert to NetworkX for spectral analysis
            G = to_networkx(data, to_undirected=True)

            if len(G.nodes) == 0:
                logger.warning("Empty graph detected, returning default features")
                return self._default_features()

            logger.debug(f"NetworkX graph created: {len(G.nodes)} nodes, {len(G.edges)} edges")

            # Compute Laplacian matrix
            edge_index, edge_weight = get_laplacian(
                data.edge_index,
                edge_weight=None,  # get_laplacian expects edge_weight, not edge_attr
                normalization='sym',
                num_nodes=data.num_nodes
            )
            logger.debug(f"Laplacian computed: {edge_index.shape[1]} non-zero entries")

            # Convert to scipy sparse matrix for eigenvalue computation
            from scipy.sparse import coo_matrix
            L = torch.sparse_coo_tensor(
                edge_index,
                edge_weight,
                torch.Size([data.num_nodes, data.num_nodes])
            ).coalesce()
            L_scipy = coo_matrix((L.values().cpu().numpy(),
                                (L.indices()[0].cpu().numpy(),
                                 L.indices()[1].cpu().numpy())),
                               shape=(data.num_nodes, data.num_nodes))

            # Compute eigenvalues
            eigenvalues = self._compute_eigenvalues(L_scipy, data.num_nodes)

            # Extract spectral features
            features = self._extract_features_from_eigenvalues(eigenvalues, data)

            return features

        except Exception as e:
            logger.warning(f"Failed to extract spectral features: {e}")
            return self._default_features()

    def _compute_eigenvalues(self, L_scipy, num_nodes: int) -> np.ndarray:
        """Compute smallest eigenvalues of Laplacian matrix."""
        k = min(self.k_eigenvalues, num_nodes - 1)

        if k <= 0:
            return np.array([0.0])

        try:
            eigenvalues, _ = eigsh(L_scipy, k=k, which='SM', sigma=0.0)
            return np.sort(eigenvalues)
        except Exception:
            # Fallback for small graphs
            L_dense = L_scipy.toarray() if hasattr(L_scipy, 'toarray') else L_scipy
            eigenvalues = np.linalg.eigvalsh(L_dense)
            return np.sort(eigenvalues)[:k]

    def _extract_features_from_eigenvalues(
        self,
        eigenvalues: np.ndarray,
        data: Data
    ) -> Dict[str, float]:
        """Extract complexity features from eigenvalue spectrum."""
        features = {}

        # Spectral gap (indicator of graph connectivity)
        if len(eigenvalues) > 1:
            features['spectral_gap'] = float(eigenvalues[1] - eigenvalues[0])
        else:
            features['spectral_gap'] = 0.0

        # Algebraic connectivity (second smallest eigenvalue)
        if len(eigenvalues) > 1:
            features['algebraic_connectivity'] = float(eigenvalues[1])
        else:
            features['algebraic_connectivity'] = 0.0

        # Chebyshev approximation order needed
        features['chebyshev_order'] = self._estimate_chebyshev_order(eigenvalues)

        # Graph size and density features
        features['num_nodes'] = float(data.num_nodes)
        features['num_edges'] = float(data.edge_index.shape[1] // 2)
        features['density'] = features['num_edges'] / (
            features['num_nodes'] * (features['num_nodes'] - 1) / 2 + 1e-8
        )

        # Spectral complexity score (higher = more complex)
        features['spectral_complexity'] = self._compute_complexity_score(features)

        return features

    def _estimate_chebyshev_order(self, eigenvalues: np.ndarray) -> float:
        """Estimate minimum Chebyshev polynomial order for spectral approximation."""
        if len(eigenvalues) < 2:
            return 1.0

        lambda_max = eigenvalues[-1]
        lambda_min = eigenvalues[0]

        # Estimate order based on spectral range
        spectral_range = lambda_max - lambda_min

        # Heuristic: order needed increases with spectral range
        order = min(
            self.chebyshev_order_max,
            max(1, int(np.log(1.0 / self.spectral_tolerance) / np.log(spectral_range + 1)))
        )

        return float(order)

    def _compute_complexity_score(self, features: Dict[str, float]) -> float:
        """Compute overall spectral complexity score."""
        # Weighted combination of spectral features
        complexity = (
            0.3 * features['chebyshev_order'] / self.chebyshev_order_max +
            0.2 * np.log(features['num_nodes'] + 1) / 10.0 +
            0.2 * features['density'] +
            0.3 * (1.0 - np.exp(-features['spectral_gap'] * 10))
        )

        return min(1.0, complexity)

    def _default_features(self) -> Dict[str, float]:
        """Return default features for edge cases."""
        return {
            'spectral_gap': 0.0,
            'algebraic_connectivity': 0.0,
            'chebyshev_order': 1.0,
            'num_nodes': 1.0,
            'num_edges': 0.0,
            'density': 0.0,
            'spectral_complexity': 0.0
        }


class MolecularGraphProcessor:
    """Processes molecular graphs with spectral augmentation and normalization."""

    def __init__(
        self,
        node_feature_dim: int = 128,
        edge_feature_dim: int = 64,
        add_self_loops: bool = True,
        normalize_features: bool = True
    ):
        """Initialize molecular graph processor.

        Args:
            node_feature_dim: Target dimension for node features
            edge_feature_dim: Target dimension for edge features
            add_self_loops: Whether to add self-loops to graphs
            normalize_features: Whether to normalize node/edge features
        """
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.add_self_loops = add_self_loops
        self.normalize_features = normalize_features
        self.spectral_extractor = SpectralFeatureExtractor()

        # Feature normalization statistics (to be computed from data)
        self.node_mean: Optional[torch.Tensor] = None
        self.node_std: Optional[torch.Tensor] = None
        self.edge_mean: Optional[torch.Tensor] = None
        self.edge_std: Optional[torch.Tensor] = None

    def process_graph(self, data: Data) -> Tuple[Data, Dict[str, float]]:
        """Process a molecular graph with spectral feature extraction.

        Args:
            data: Raw PyTorch Geometric data object

        Returns:
            Tuple of (processed_data, spectral_features)
        """
        # Extract spectral features first
        spectral_features = self.spectral_extractor.extract_spectral_features(data)

        # Process graph structure
        processed_data = self._process_structure(data.clone())

        # Process features
        processed_data = self._process_features(processed_data)

        # Add spectral features as graph-level attributes
        for key, value in spectral_features.items():
            setattr(processed_data, f'spectral_{key}', torch.tensor(value))

        return processed_data, spectral_features

    def _process_structure(self, data: Data) -> Data:
        """Process graph structure (add self-loops, etc.)."""
        if self.add_self_loops and hasattr(data, 'edge_index'):
            # Add self-loops
            num_nodes = data.num_nodes
            self_loop_edge_index = torch.stack([
                torch.arange(num_nodes),
                torch.arange(num_nodes)
            ])

            data.edge_index = torch.cat([data.edge_index, self_loop_edge_index], dim=1)

            # Add self-loop edge features (zeros)
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                self_loop_attr = torch.zeros(
                    num_nodes, data.edge_attr.shape[1],
                    dtype=data.edge_attr.dtype
                )
                data.edge_attr = torch.cat([data.edge_attr, self_loop_attr], dim=0)

        return data

    def _process_features(self, data: Data) -> Data:
        """Process and normalize node/edge features."""
        # Normalize BEFORE resizing so dimensions match the raw feature stats
        if hasattr(data, 'x') and data.x is not None:
            if self.normalize_features and self.node_mean is not None:
                raw_dim = data.x.shape[-1]
                if self.node_mean.shape[0] == raw_dim:
                    data.x = (data.x - self.node_mean) / (self.node_std + 1e-8)

            data.x = self._resize_features(data.x, self.node_feature_dim)

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if self.normalize_features and self.edge_mean is not None:
                raw_dim = data.edge_attr.shape[-1]
                if self.edge_mean.shape[0] == raw_dim:
                    data.edge_attr = (data.edge_attr - self.edge_mean) / (self.edge_std + 1e-8)

            data.edge_attr = self._resize_features(data.edge_attr, self.edge_feature_dim)

        return data

    def _resize_features(self, features: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Resize features to target dimension."""
        current_dim = features.shape[-1]

        if current_dim == target_dim:
            return features
        elif current_dim < target_dim:
            # Pad with zeros
            padding = torch.zeros(
                features.shape[:-1] + (target_dim - current_dim,),
                dtype=features.dtype,
                device=features.device
            )
            return torch.cat([features, padding], dim=-1)
        else:
            # Truncate
            return features[..., :target_dim]

    def fit_normalizer(self, data_list: List[Data]) -> None:
        """Compute normalization statistics from training data.

        Args:
            data_list: List of training graph data
        """
        if not self.normalize_features:
            return

        logger.info("Computing feature normalization statistics...")

        all_node_features = []
        all_edge_features = []

        for data in data_list:
            if hasattr(data, 'x') and data.x is not None:
                all_node_features.append(data.x)
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                all_edge_features.append(data.edge_attr)

        # Compute node feature statistics
        if all_node_features:
            all_node_features = torch.cat(all_node_features, dim=0)
            self.node_mean = all_node_features.mean(dim=0)
            self.node_std = all_node_features.std(dim=0)

        # Compute edge feature statistics
        if all_edge_features:
            all_edge_features = torch.cat(all_edge_features, dim=0)
            self.edge_mean = all_edge_features.mean(dim=0)
            self.edge_std = all_edge_features.std(dim=0)

        logger.info("Feature normalization statistics computed.")