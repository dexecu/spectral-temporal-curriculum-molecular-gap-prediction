"""Data loading utilities for PCQM4Mv2 with spectral curriculum learning."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
# Patch torch.load for OGB compatibility with PyTorch 2.6+ (weights_only default change)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeomDataLoader
import pytorch_lightning as pl

try:
    from ogb.lsc import PCQM4Mv2Dataset
    from ogb.utils.mol import smiles2graph as _ogb_smiles2graph

    def safe_smiles2graph(smiles):
        """Wrapper that returns a dummy graph for invalid SMILES instead of crashing."""
        try:
            return _ogb_smiles2graph(smiles)
        except Exception:
            # Return minimal valid graph for unparseable molecules
            return {
                'edge_index': np.zeros((2, 0), dtype=np.int64),
                'edge_feat': np.zeros((0, 3), dtype=np.int64),
                'node_feat': np.zeros((1, 9), dtype=np.int64),
                'num_nodes': 1,
            }
except ImportError:
    PCQM4Mv2Dataset = None
    safe_smiles2graph = None
    logging.warning("OGB not installed. Using mock dataset for development.")

from .preprocessing import MolecularGraphProcessor, SpectralFeatureExtractor

logger = logging.getLogger(__name__)


class MockPCQM4Mv2Dataset:
    """Mock dataset for development when OGB is not available."""

    def __init__(self, root: str = "data", subset: bool = True):
        self.root = root
        self.subset = subset
        self.num_samples = 1000 if subset else 100000

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Data:
        # Generate synthetic molecular graph
        num_nodes = np.random.randint(5, 50)
        num_edges = np.random.randint(num_nodes, num_nodes * 3)

        # Random edge indices
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Random node features (atomic features)
        x = torch.randn(num_nodes, 9)  # Common atomic feature size

        # Random edge features
        edge_attr = torch.randn(num_edges, 3)

        # Random HOMO-LUMO gap target (in eV)
        y = torch.tensor([np.random.uniform(1.0, 10.0)])

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_nodes
        )

    def get_idx_split(self) -> Dict[str, np.ndarray]:
        """Return train/val/test splits."""
        total = self.num_samples
        train_size = int(0.8 * total)
        val_size = int(0.1 * total)

        return {
            'train': np.arange(train_size),
            'valid': np.arange(train_size, train_size + val_size),
            'test': np.arange(train_size + val_size, total)
        }


class SpectralComplexityDataset(Dataset):
    """Dataset wrapper that adds spectral complexity-based sorting and curriculum."""

    def __init__(
        self,
        base_dataset: Dataset,
        processor: MolecularGraphProcessor,
        cache_spectral_features: bool = True,
        cache_file: Optional[str] = None
    ):
        """Initialize spectral complexity dataset.

        Args:
            base_dataset: Base molecular dataset
            processor: Graph processor with spectral feature extraction
            cache_spectral_features: Whether to cache computed spectral features
            cache_file: Path to cache file for spectral features
        """
        self.base_dataset = base_dataset
        self.processor = processor
        self.cache_spectral_features = cache_spectral_features
        self.cache_file = cache_file

        # Cache for spectral features
        self.spectral_features_cache: Dict[int, Dict[str, float]] = {}

        # Load cached features if available
        if self.cache_file and Path(self.cache_file).exists():
            self._load_cache()

        # Complexity-based indices for curriculum learning
        self.complexity_sorted_indices: Optional[List[int]] = None

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[Data, Dict[str, float]]:
        """Get processed graph data with spectral features.

        Args:
            idx: Sample index

        Returns:
            Tuple of (processed_graph_data, spectral_features)
        """
        # Get base data
        data = self.base_dataset[idx]

        # Check cache first
        if idx in self.spectral_features_cache:
            spectral_features = self.spectral_features_cache[idx]
            # Still need to process the graph structure
            processed_data, _ = self.processor.process_graph(data)

            # Add cached spectral features
            for key, value in spectral_features.items():
                setattr(processed_data, f'spectral_{key}', torch.tensor(value))

        else:
            # Process graph and compute spectral features
            processed_data, spectral_features = self.processor.process_graph(data)

            # Cache spectral features
            if self.cache_spectral_features:
                self.spectral_features_cache[idx] = spectral_features

        return processed_data, spectral_features

    def compute_all_spectral_features(self, max_samples: Optional[int] = None) -> None:
        """Pre-compute spectral features for all samples.

        Args:
            max_samples: Maximum number of samples to process (for development)
        """
        logger.info("Computing spectral features for curriculum learning...")

        n_samples = min(len(self), max_samples) if max_samples else len(self)

        for idx in range(n_samples):
            if idx % 1000 == 0:
                logger.info(f"Processed {idx}/{n_samples} samples")

            if idx not in self.spectral_features_cache:
                data = self.base_dataset[idx]
                _, spectral_features = self.processor.process_graph(data)

                if self.cache_spectral_features:
                    self.spectral_features_cache[idx] = spectral_features

        logger.info(f"Computed spectral features for {len(self.spectral_features_cache)} samples")

        # Save cache
        if self.cache_file:
            self._save_cache()

    def get_curriculum_order(
        self,
        curriculum_strategy: str = "spectral_complexity",
        reverse: bool = False
    ) -> List[int]:
        """Get sample indices sorted by complexity for curriculum learning.

        Args:
            curriculum_strategy: Strategy for curriculum ordering
            reverse: Whether to reverse the order (hard to easy)

        Returns:
            List of sample indices in curriculum order
        """
        if not self.spectral_features_cache:
            logger.warning("No spectral features computed. Computing now...")
            self.compute_all_spectral_features()

        # Extract complexity scores
        complexity_scores = []
        valid_indices = []

        for idx in range(len(self)):
            if idx in self.spectral_features_cache:
                features = self.spectral_features_cache[idx]

                if curriculum_strategy == "spectral_complexity":
                    score = features.get('spectral_complexity', 0.0)
                elif curriculum_strategy == "graph_size":
                    score = features.get('num_nodes', 0.0)
                elif curriculum_strategy == "spectral_gap":
                    score = features.get('spectral_gap', 0.0)
                elif curriculum_strategy == "chebyshev_order":
                    score = features.get('chebyshev_order', 1.0)
                else:
                    score = features.get('spectral_complexity', 0.0)

                complexity_scores.append((score, idx))
                valid_indices.append(idx)

        # Sort by complexity
        complexity_scores.sort(key=lambda x: x[0], reverse=reverse)
        sorted_indices = [idx for _, idx in complexity_scores]

        self.complexity_sorted_indices = sorted_indices
        return sorted_indices

    def get_curriculum_subset(
        self,
        fraction: float,
        curriculum_strategy: str = "spectral_complexity"
    ) -> 'SpectralComplexityDataset':
        """Get a subset of data based on curriculum difficulty.

        Args:
            fraction: Fraction of data to include (0.0 to 1.0)
            curriculum_strategy: Strategy for curriculum ordering

        Returns:
            New dataset with subset of samples
        """
        if self.complexity_sorted_indices is None:
            self.get_curriculum_order(curriculum_strategy)

        n_samples = int(len(self.complexity_sorted_indices) * fraction)
        subset_indices = self.complexity_sorted_indices[:n_samples]

        # Create subset of base dataset
        base_subset = Subset(self.base_dataset, subset_indices)

        # Create new SpectralComplexityDataset with subset
        subset_dataset = SpectralComplexityDataset(
            base_dataset=base_subset,
            processor=self.processor,
            cache_spectral_features=False  # Use parent cache
        )

        # Share spectral features cache
        subset_dataset.spectral_features_cache = {
            i: self.spectral_features_cache[idx]
            for i, idx in enumerate(subset_indices)
            if idx in self.spectral_features_cache
        }

        return subset_dataset

    def _save_cache(self) -> None:
        """Save spectral features cache to disk."""
        if self.cache_file:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.spectral_features_cache, f)
                logger.info(f"Saved spectral features cache to {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    def _load_cache(self) -> None:
        """Load spectral features cache from disk."""
        try:
            with open(self.cache_file, 'rb') as f:
                self.spectral_features_cache = pickle.load(f)
            logger.info(f"Loaded {len(self.spectral_features_cache)} cached spectral features")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.spectral_features_cache = {}


class PCQM4Mv2DataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for PCQM4Mv2 with spectral curriculum learning."""

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        subset: bool = True,
        curriculum_strategy: str = "spectral_complexity",
        max_samples: Optional[int] = None,
        node_feature_dim: int = 128,
        edge_feature_dim: int = 64,
        cache_spectral_features: bool = True
    ):
        """Initialize PCQM4Mv2 data module.

        Args:
            data_dir: Directory to store/load data
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for GPU transfer
            subset: Whether to use subset of data for development
            curriculum_strategy: Strategy for curriculum learning
            max_samples: Maximum samples to use (for development)
            node_feature_dim: Target node feature dimension
            edge_feature_dim: Target edge feature dimension
            cache_spectral_features: Whether to cache spectral features
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.subset = subset
        self.curriculum_strategy = curriculum_strategy
        self.max_samples = max_samples
        self.cache_spectral_features = cache_spectral_features

        # Initialize processor
        self.processor = MolecularGraphProcessor(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim
        )

        # Datasets
        self.dataset_train: Optional[SpectralComplexityDataset] = None
        self.dataset_val: Optional[SpectralComplexityDataset] = None
        self.dataset_test: Optional[SpectralComplexityDataset] = None

        # Cache files
        self.cache_dir = self.data_dir / "spectral_cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def prepare_data(self) -> None:
        """Download and prepare the dataset."""
        # Create data directory
        self.data_dir.mkdir(exist_ok=True, parents=True)

        # Initialize base dataset with safe SMILES parser
        if PCQM4Mv2Dataset is not None:
            try:
                _ = PCQM4Mv2Dataset(root=str(self.data_dir), smiles2graph=safe_smiles2graph)
                logger.info("PCQM4Mv2 dataset initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PCQM4Mv2: {e}")
        else:
            logger.info("Using mock dataset for development")

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for training, validation, and testing.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        # Initialize base dataset with safe SMILES parser
        if PCQM4Mv2Dataset is not None:
            try:
                base_dataset = PCQM4Mv2Dataset(root=str(self.data_dir), smiles2graph=safe_smiles2graph)
            except Exception:
                logger.warning("Using mock dataset due to OGB initialization failure")
                base_dataset = MockPCQM4Mv2Dataset(root=str(self.data_dir), subset=self.subset)
        else:
            base_dataset = MockPCQM4Mv2Dataset(root=str(self.data_dir), subset=self.subset)

        # Get data splits
        split_idx = base_dataset.get_idx_split()

        # Limit samples if specified
        if self.max_samples:
            for split in split_idx:
                split_idx[split] = split_idx[split][:min(len(split_idx[split]), self.max_samples)]

        if stage in (None, 'fit'):
            # Training dataset
            train_subset = Subset(base_dataset, split_idx['train'])
            self.dataset_train = SpectralComplexityDataset(
                base_dataset=train_subset,
                processor=self.processor,
                cache_spectral_features=self.cache_spectral_features,
                cache_file=str(self.cache_dir / "train_spectral_features.pkl")
            )

            # Validation dataset
            val_subset = Subset(base_dataset, split_idx['valid'])
            self.dataset_val = SpectralComplexityDataset(
                base_dataset=val_subset,
                processor=self.processor,
                cache_spectral_features=self.cache_spectral_features,
                cache_file=str(self.cache_dir / "val_spectral_features.pkl")
            )

            # Compute spectral features for curriculum learning
            logger.info("Computing spectral features for curriculum learning...")
            self.dataset_train.compute_all_spectral_features()
            self.dataset_val.compute_all_spectral_features()

            # Fit normalizer on training data
            logger.info("Fitting feature normalizer...")
            train_data_list = [self.dataset_train.base_dataset[i] for i in range(min(1000, len(self.dataset_train)))]
            self.processor.fit_normalizer(train_data_list)

        if stage in (None, 'test'):
            # Test dataset
            test_subset = Subset(base_dataset, split_idx['test'])
            self.dataset_test = SpectralComplexityDataset(
                base_dataset=test_subset,
                processor=self.processor,
                cache_spectral_features=self.cache_spectral_features,
                cache_file=str(self.cache_dir / "test_spectral_features.pkl")
            )

            self.dataset_test.compute_all_spectral_features()

    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        return GeomDataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        return GeomDataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        return GeomDataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )

    def get_curriculum_dataloader(
        self,
        stage: str,
        difficulty_fraction: float,
        shuffle: bool = True
    ) -> DataLoader:
        """Get data loader with curriculum-based subset.

        Args:
            stage: 'train', 'val', or 'test'
            difficulty_fraction: Fraction of easiest samples to include
            shuffle: Whether to shuffle the data

        Returns:
            Data loader with curriculum subset
        """
        if stage == 'train':
            dataset = self.dataset_train
        elif stage == 'val':
            dataset = self.dataset_val
        else:
            dataset = self.dataset_test

        if dataset is None:
            raise ValueError(f"Dataset for stage '{stage}' not initialized")

        # Get curriculum subset
        curriculum_dataset = dataset.get_curriculum_subset(
            fraction=difficulty_fraction,
            curriculum_strategy=self.curriculum_strategy
        )

        return GeomDataLoader(
            curriculum_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )