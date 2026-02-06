"""Spectral-Temporal Curriculum Molecular Gap Prediction.

A research project that introduces a spectral-aware curriculum learning strategy
for HOMO-LUMO gap prediction on PCQM4Mv2.
"""

__version__ = "0.1.0"
__author__ = "A-SHOJAEI"

from . import data, models, training, evaluation, utils

__all__ = ["data", "models", "training", "evaluation", "utils"]