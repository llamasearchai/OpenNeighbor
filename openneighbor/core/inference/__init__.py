"""
Inference module for OpenNeighbor.

Author: Nik Jois <nikjois@llamasearch.ai>

This module provides inference functionality for OpenNeighbor models,
including predictors, batch inference, and real-time serving capabilities.
"""

from .predictor import OpenNeighborPredictor

__all__ = ['OpenNeighborPredictor'] 