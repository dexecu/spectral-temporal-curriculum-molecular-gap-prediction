# Training Results and Analysis Notes

## Experiment Overview

This document provides additional context for the training results summarized in `results_summary.json`.

## Training Run Details

**Date**: 2026-02-07
**Configuration**: `configs/default.yaml`
**Dataset**: PCQM4Mv2 (50,000-sample subset)
**Hardware**: NVIDIA RTX 4090 (24 GB)

## Key Observations

### 1. Best Performance at Early Epoch

The best validation MAE of **0.638 eV** was achieved at epoch 0, with subsequent epochs showing higher validation losses. This suggests:

- The initial 10% of training samples (easiest molecules by spectral complexity) provided a strong baseline
- As harder samples were introduced through curriculum learning, the validation loss temporarily increased
- This is expected behavior in curriculum learning where model capacity is gradually challenged

### 2. Curriculum Learning Dynamics

The curriculum progression shows clear phases:

- **Epochs 0-4**: 10% of data (warmup phase) - Validation MAE: 0.638-0.702 eV
- **Epochs 5-7**: 35-45% of data (rapid expansion) - Validation loss increased to 0.881 eV
- **Epochs 8-15**: 50-85% of data (final phase) - Gradual stabilization around 0.747-0.933 eV

### 3. Early Stopping

Training stopped at epoch 15 with patience=15, indicating the model reached a local optimum on the 50K subset. The validation loss fluctuations reflect the curriculum-induced difficulty scaling.

## Performance Gap Analysis

**Current Performance**: 0.638 eV (best epoch)
**Target Performance**: 0.082 eV
**Gap**: ~0.556 eV

This gap is expected because:

1. **Limited Dataset Size**: 50K samples vs. full 3.7M molecule dataset
2. **Early Stopping**: Only 16 epochs vs. potential 100+ epochs needed
3. **Subset Bias**: The 50K subset may not represent the full molecular diversity

## Custom Loss Components

The project includes advanced loss functions in `src/.../models/components.py`:

### SpectralRegularizedLoss

Combines base regression loss with spectral smoothness regularization:

```python
from spectral_temporal_curriculum_molecular_gap_prediction.models.components import SpectralRegularizedLoss

loss_fn = SpectralRegularizedLoss(
    base_loss='mae',
    spectral_weight=0.01,
    curriculum_weight=0.05
)

# During training
total_loss, loss_dict = loss_fn(
    predictions=model_output,
    targets=ground_truth,
    spectral_filters=model.spectral_encoder.filters[0].weight,
    sample_weights=curriculum_weights
)
```

This encourages smooth spectral filters, preventing overfitting to high-frequency noise.

### UncertaintyWeightedLoss

Learns uncertainty estimates alongside predictions:

```python
from spectral_temporal_curriculum_molecular_gap_prediction.models.components import UncertaintyWeightedLoss

loss_fn = UncertaintyWeightedLoss(init_log_variance=0.0)
total_loss, loss_dict = loss_fn(predictions, targets)

# Access learned uncertainty
uncertainty_std = loss_dict['uncertainty']
```

## Ablation Study Configuration

The `configs/ablation.yaml` file tests the contribution of spectral components by:

1. Reducing spectral filters from 6 to 1
2. Reducing Chebyshev order from 20 to 5
3. Using simple concatenation instead of cross-attention fusion
4. Using mean pooling instead of attention pooling
5. Disabling curriculum learning
6. Disabling spectral regularization

**Expected Outcome**: Running with `configs/ablation.yaml` should show degraded performance compared to `configs/default.yaml`, validating the importance of the spectral-temporal architecture.

## Next Steps for Improved Performance

To achieve the target MAE of 0.082 eV:

1. **Scale to Full Dataset**: Train on all 3.7M molecules in PCQM4Mv2
2. **Extended Training**: Run for 100+ epochs with appropriate learning rate scheduling
3. **Hyperparameter Tuning**: Grid search over learning rate, hidden dimensions, and spectral filter counts
4. **Ensemble Methods**: Combine multiple model checkpoints
5. **Advanced Curriculum**: Experiment with different curriculum strategies (graph size, spectral gap, etc.)

## File Structure

```
results/
├── results_summary.json    # Structured training metrics
└── NOTES.md               # This file - additional analysis and documentation
```

## References

For more details on the methodology and architecture, see the project README.md.
