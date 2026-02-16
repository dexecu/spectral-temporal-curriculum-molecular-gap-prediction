"""Prediction script for trained spectral-temporal models.

This script loads a trained model checkpoint and performs predictions on
molecular graphs. It supports both single molecule prediction and batch
prediction modes.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import numpy as np
from torch_geometric.data import Data, Batch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from spectral_temporal_curriculum_molecular_gap_prediction.training.trainer import CurriculumTrainer
from spectral_temporal_curriculum_molecular_gap_prediction.data.loader import PCQM4Mv2DataModule
from spectral_temporal_curriculum_molecular_gap_prediction.data.preprocessing import MolecularFeatureExtractor

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration.

    Args:
        verbose: If True, set logging level to DEBUG
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def load_model(checkpoint_path: str, device: str = 'cpu') -> CurriculumTrainer:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint file
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded model in evaluation mode

    Raises:
        FileNotFoundError: If checkpoint file does not exist
        RuntimeError: If model loading fails
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading model from {checkpoint_path}")

    try:
        # Load checkpoint
        model = CurriculumTrainer.load_from_checkpoint(
            checkpoint_path,
            map_location=device
        )
        model.eval()
        model.to(device)

        logger.info(f"Model loaded successfully on {device}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}") from e


def create_sample_molecule() -> Data:
    """Create a sample molecular graph for demonstration.

    Returns:
        PyTorch Geometric Data object representing a benzene molecule
    """
    # Benzene molecule (C6H6) - simplified representation
    # 6 carbon atoms in a ring
    num_nodes = 6

    # Create ring connectivity
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],  # source nodes
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]   # target nodes
    ], dtype=torch.long)

    # Random node features (in practice, these would be atomic features)
    x = torch.randn(num_nodes, 128)

    # Random edge features
    edge_attr = torch.randn(edge_index.size(1), 64)

    # Mock HOMO-LUMO gap (eV)
    y = torch.tensor([5.5], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    logger.info(f"Created sample molecule with {num_nodes} atoms and {edge_index.size(1)} bonds")

    return data


def predict_single(
    model: CurriculumTrainer,
    molecular_data: Data,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Predict HOMO-LUMO gap for a single molecule.

    Args:
        model: Trained model
        molecular_data: Molecular graph data
        device: Device for computation

    Returns:
        Dictionary containing prediction and confidence metrics
    """
    molecular_data = molecular_data.to(device)

    with torch.no_grad():
        prediction = model(molecular_data)

    result = {
        'homo_lumo_gap_ev': prediction.item(),
        'confidence': 1.0,  # Placeholder for uncertainty estimation
    }

    return result


def predict_batch(
    model: CurriculumTrainer,
    molecular_data_list: List[Data],
    device: str = 'cpu',
    batch_size: int = 32
) -> List[Dict[str, float]]:
    """Predict HOMO-LUMO gaps for multiple molecules.

    Args:
        model: Trained model
        molecular_data_list: List of molecular graph data
        device: Device for computation
        batch_size: Batch size for processing

    Returns:
        List of prediction dictionaries
    """
    results = []

    for i in range(0, len(molecular_data_list), batch_size):
        batch_data = molecular_data_list[i:i + batch_size]
        batch = Batch.from_data_list(batch_data).to(device)

        with torch.no_grad():
            predictions = model(batch)

        for j, pred in enumerate(predictions):
            results.append({
                'molecule_idx': i + j,
                'homo_lumo_gap_ev': pred.item(),
                'confidence': 1.0,
            })

    logger.info(f"Predicted HOMO-LUMO gaps for {len(molecular_data_list)} molecules")

    return results


def predict_from_dataset(
    model: CurriculumTrainer,
    data_dir: str,
    split: str = 'test',
    num_samples: Optional[int] = None,
    device: str = 'cpu'
) -> List[Dict[str, float]]:
    """Predict on molecules from PCQM4Mv2 dataset.

    Args:
        model: Trained model
        data_dir: Path to dataset directory
        split: Dataset split ('train', 'val', 'test')
        num_samples: Number of samples to predict (None for all)
        device: Device for computation

    Returns:
        List of prediction dictionaries
    """
    logger.info(f"Loading {split} split from {data_dir}")

    # Initialize data module
    datamodule = PCQM4Mv2DataModule(
        data_dir=data_dir,
        batch_size=32,
        num_workers=0,
        subset=True,
        max_samples=num_samples or 100
    )
    datamodule.setup(split)

    # Get appropriate dataloader
    if split == 'train':
        dataloader = datamodule.train_dataloader()
    elif split == 'val':
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.test_dataloader()

    results = []

    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            predictions = model(batch)

        # Extract ground truth if available
        y_true = batch.y if hasattr(batch, 'y') else None

        for i, pred in enumerate(predictions):
            result = {
                'batch': batch_idx,
                'sample_idx': i,
                'predicted_gap_ev': pred.item(),
            }

            if y_true is not None and i < len(y_true):
                result['true_gap_ev'] = y_true[i].item()
                result['absolute_error_ev'] = abs(pred.item() - y_true[i].item())

            results.append(result)

        if num_samples and len(results) >= num_samples:
            results = results[:num_samples]
            break

    logger.info(f"Completed predictions on {len(results)} samples")

    return results


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(
        description='Predict molecular HOMO-LUMO gaps using trained model'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='checkpoints/best_model.ckpt',
        help='Path to trained model checkpoint'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='sample',
        help='Input type: "sample" for demo molecule, "dataset" for PCQM4Mv2'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to dataset directory (for dataset input)'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to predict on'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to predict'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device for computation'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save predictions (optional)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info(f"Using device: {device}")

    # Check if model checkpoint exists
    if not os.path.exists(args.model_path):
        logger.warning(f"Model checkpoint not found at {args.model_path}")
        logger.info("Attempting to find a checkpoint in checkpoints/ directory...")

        checkpoint_dir = Path(args.model_path).parent
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if checkpoints:
                args.model_path = str(checkpoints[0])
                logger.info(f"Using checkpoint: {args.model_path}")
            else:
                logger.error("No checkpoint files found. Please train a model first.")
                sys.exit(1)
        else:
            logger.error("Checkpoints directory not found. Please train a model first.")
            sys.exit(1)

    # Load model
    try:
        model = load_model(args.model_path, device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Perform prediction based on input type
    if args.input == 'sample':
        logger.info("Generating sample molecule for prediction...")
        sample_molecule = create_sample_molecule()

        result = predict_single(model, sample_molecule, device)

        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Predicted HOMO-LUMO Gap: {result['homo_lumo_gap_ev']:.4f} eV")
        print(f"Confidence Score: {result['confidence']:.2f}")
        print("="*60 + "\n")

    elif args.input == 'dataset':
        logger.info(f"Predicting on {args.split} split from dataset...")

        results = predict_from_dataset(
            model=model,
            data_dir=args.data_dir,
            split=args.split,
            num_samples=args.num_samples,
            device=device
        )

        # Compute statistics
        predictions = [r['predicted_gap_ev'] for r in results]

        print("\n" + "="*60)
        print(f"PREDICTION STATISTICS ({len(results)} samples)")
        print("="*60)
        print(f"Mean predicted gap: {np.mean(predictions):.4f} eV")
        print(f"Std predicted gap: {np.std(predictions):.4f} eV")
        print(f"Min predicted gap: {np.min(predictions):.4f} eV")
        print(f"Max predicted gap: {np.max(predictions):.4f} eV")

        # If ground truth available, compute metrics
        if 'true_gap_ev' in results[0]:
            errors = [r['absolute_error_ev'] for r in results]
            print("\n" + "-"*60)
            print("EVALUATION METRICS")
            print("-"*60)
            print(f"Mean Absolute Error: {np.mean(errors):.4f} eV")
            print(f"RMSE: {np.sqrt(np.mean([e**2 for e in errors])):.4f} eV")

        print("="*60 + "\n")

        # Show first few predictions
        print("First 5 predictions:")
        for i, result in enumerate(results[:5]):
            print(f"  Sample {i+1}: {result['predicted_gap_ev']:.4f} eV", end='')
            if 'true_gap_ev' in result:
                print(f" (true: {result['true_gap_ev']:.4f} eV, error: {result['absolute_error_ev']:.4f} eV)")
            else:
                print()

        # Save results if output path specified
        if args.output:
            import json
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"Predictions saved to {args.output}")

    else:
        logger.error(f"Unknown input type: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
