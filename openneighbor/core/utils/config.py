"""
Configuration management for OpenNeighbor.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration management class for OpenNeighbor."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration."""
        self._config = config_dict or {}
        self._defaults = self._get_default_config()
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'Config':
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls(config_dict)
    
    @classmethod
    def from_env(cls, prefix: str = 'OPENNEIGHBOR_') -> 'Config':
        """Load configuration from environment variables."""
        config_dict = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # Try to parse as JSON, fallback to string
                try:
                    config_dict[config_key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    config_dict[config_key] = value
        
        return cls(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            # Try defaults
            value = self._defaults
            try:
                for k in keys:
                    value = value[k]
                return value
            except (KeyError, TypeError):
                return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value with dot notation support."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, other: Union[Dict[str, Any], 'Config']) -> None:
        """Update configuration with another config or dict."""
        if isinstance(other, Config):
            other_dict = other._config
        else:
            other_dict = other
        
        self._deep_update(self._config, other_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()
    
    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def _deep_update(self, base_dict: Dict[str, Any], 
                     update_dict: Dict[str, Any]) -> None:
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'model': {
                'name': 'openneighbor',
                'hidden_dim': 768,
                'num_attention_heads': 8,
                'num_layers': 6,
                'dropout': 0.1,
                'activation': 'gelu',
                'num_users': 100000,
                'num_items': 500000,
                'num_content_types': 20,
                'num_categories': 50,
                'spatial_heads': 8,
                'spatial_layers': 2,
                'gnn_layers': 3,
                'gnn_type': 'gat',
                'text_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'freeze_text_encoder': False,
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'num_epochs': 100,
                'warmup_steps': 1000,
                'gradient_clipping': 1.0,
                'early_stopping_patience': 10,
                'checkpoint_every': 5,
                'use_wandb': False,
                'wandb_project': 'openneighbor',
                'mixed_precision': True,
                'distributed_training': False,
            },
            'data': {
                'max_sequence_length': 512,
                'max_neighbors': 50,
                'spatial_radius_km': 5.0,
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1,
                'num_synthetic_users': 10000,
                'num_synthetic_items': 50000,
            },
            'inference': {
                'batch_size': 64,
                'top_k': 10,
                'diversity_threshold': 0.7,
                'diversity_weight': 0.1,
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 1,
                'timeout': 30,
                'max_request_size': 1024 * 1024,  # 1MB
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': None,
            },
            'monitoring': {
                'enable_metrics': True,
                'metrics_port': 9090,
                'enable_tracing': False,
            }
        }
    
    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Support dictionary-style setting."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return self.get(key) is not None
    
    def __str__(self) -> str:
        """String representation."""
        return json.dumps(self._config, indent=2)
    
    def __repr__(self) -> str:
        """Representation."""
        return f"Config({json.dumps(self._config, indent=2)})"

def validate_config(config: Union[Config, Dict[str, Any]]) -> List[str]:
    """Validate configuration and return list of errors."""
    if isinstance(config, Config):
        config_dict = config.to_dict()
    else:
        config_dict = config
    
    errors = []
    
    # Required fields
    required_fields = [
        'model.hidden_dim',
        'model.num_users',
        'model.num_items',
        'training.batch_size',
        'training.learning_rate',
        'training.num_epochs',
    ]
    
    for field in required_fields:
        keys = field.split('.')
        value = config_dict
        try:
            for key in keys:
                value = value[key]
            if value is None:
                errors.append(f"Required field '{field}' is None")
        except (KeyError, TypeError):
            errors.append(f"Required field '{field}' is missing")
    
    # Validate ranges
    validations = [
        ('model.hidden_dim', lambda x: isinstance(x, int) and x > 0, 
         "hidden_dim must be a positive integer"),
        ('model.dropout', lambda x: isinstance(x, (int, float)) and 0 <= x <= 1, 
         "dropout must be between 0 and 1"),
        ('training.batch_size', lambda x: isinstance(x, int) and x > 0, 
         "batch_size must be a positive integer"),
        ('training.learning_rate', lambda x: isinstance(x, (int, float)) and x > 0, 
         "learning_rate must be positive"),
        ('training.num_epochs', lambda x: isinstance(x, int) and x > 0, 
         "num_epochs must be a positive integer"),
    ]
    
    for field, validator, error_msg in validations:
        keys = field.split('.')
        value = config_dict
        try:
            for key in keys:
                value = value[key]
            if not validator(value):
                errors.append(f"{field}: {error_msg}")
        except (KeyError, TypeError):
            pass  # Already caught by required fields check
    
    return errors 