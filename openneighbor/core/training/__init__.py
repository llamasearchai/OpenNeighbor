"""
Training module for OpenNeighbor.

Author: Nik Jois <nikjois@llamasearch.ai>

This module provides comprehensive training functionality for OpenNeighbor models,
including trainers, metrics, losses, and callbacks.
"""

from .trainer import OpenNeighborTrainer

__all__ = ['OpenNeighborTrainer'] 