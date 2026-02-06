"""Training modules with curriculum learning."""

from .trainer import (
    CurriculumTrainer,
    CurriculumScheduler,
    SpectralComplexityScheduler,
    CustomLoss
)

__all__ = [
    "CurriculumTrainer",
    "CurriculumScheduler",
    "SpectralComplexityScheduler",
    "CustomLoss"
]