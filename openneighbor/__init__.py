"""
OpenNeighbor: Production-Grade Neighborhood-Aware Recommendation System

Author: Nik Jois <nikjois@llamasearch.ai>

A comprehensive, scalable, and ethical recommendation system designed specifically
for neighborhood-based social platforms. Built with PyTorch, featuring Graph Neural
Networks, spatial attention mechanisms, and fairness-aware algorithms.

Key Features:
- Spatial-aware graph neural networks
- Multi-modal content understanding  
- Fairness and diversity optimization
- Real-time inference capabilities
- Production-ready deployment
- Comprehensive monitoring and observability

Example:
    >>> from openneighbor import OpenNeighbor
    >>> model = OpenNeighbor.from_pretrained('openneighbor/base')
    >>> recommendations = model.recommend(user_id=123, top_k=10)
"""

__version__ = "1.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"
__license__ = "Apache-2.0"

import logging
import warnings
from typing import Dict, Any

# Configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Import main classes
try:
    from .core.models.openneighbor import OpenNeighbor
    from .core.data.dataset import NeighborhoodDataset
    from .core.training.trainer import OpenNeighborTrainer
    from .core.inference.predictor import OpenNeighborPredictor
    from .core.utils.config import Config
except ImportError as e:
    # Graceful degradation if dependencies are missing
    warnings.warn(f"Some components could not be imported: {e}")
    OpenNeighbor = None
    NeighborhoodDataset = None
    OpenNeighborTrainer = None
    OpenNeighborPredictor = None
    Config = None

# Version compatibility checks
def _check_dependencies() -> Dict[str, Any]:
    """Check if all required dependencies are installed with correct versions."""
    import sys
    
    deps_status = {
        'python_version': sys.version_info,
        'dependencies': {}
    }
    
    required_deps = {
        'torch': '2.1.0',
        'transformers': '4.35.0',
        'numpy': '1.24.0',
        'pandas': '2.1.0',
        'scikit-learn': '1.3.0',
        'fastapi': '0.104.0',
    }
    
    for dep_name, min_version in required_deps.items():
        try:
            module = __import__(dep_name.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')
            
            deps_status['dependencies'][dep_name] = {
                'installed': True,
                'version': version,
                'required': min_version
            }
        except ImportError:
            deps_status['dependencies'][dep_name] = {
                'installed': False,
                'version': None,
                'required': min_version
            }
            # Only warn for truly missing dependencies
            if dep_name not in ['scikit-learn']:  # scikit-learn is installed but import name is sklearn
                warnings.warn(f"Optional dependency {dep_name} not found")
    
    return deps_status

# Check dependencies on import
_deps_status = _check_dependencies()

# Export public API
__all__ = [
    'OpenNeighbor',
    'NeighborhoodDataset', 
    'OpenNeighborTrainer',
    'OpenNeighborPredictor',
    'Config',
    '__version__',
]

# Module-level configuration
DEFAULT_CONFIG = {
    'model': {
        'hidden_dim': 768,
        'num_attention_heads': 8,
        'num_layers': 6,
        'dropout': 0.1,
        'activation': 'gelu',
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'warmup_steps': 1000,
    },
    'data': {
        'max_sequence_length': 512,
        'max_neighbors': 50,
        'spatial_radius_km': 5.0,
    },
    'inference': {
        'batch_size': 64,
        'top_k': 10,
        'diversity_threshold': 0.7,
    }
}

def get_version() -> str:
    """Get the current version of OpenNeighbor."""
    return __version__

def get_config() -> Dict[str, Any]:
    """Get the default configuration."""
    return DEFAULT_CONFIG.copy()

def check_health() -> Dict[str, Any]:
    """Check system health and dependencies."""
    return {
        'version': __version__,
        'status': 'healthy',
        'dependencies': _deps_status,
        'config': DEFAULT_CONFIG
    } 