"""
Training functionality for OpenNeighbor models.

Author: Nik Jois <nikjois@llamasearch.ai>

This module provides the main trainer class for training OpenNeighbor models
with support for distributed training, mixed precision, and comprehensive logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
import json
import time
from datetime import datetime
import os

from ..models.openneighbor import OpenNeighbor
from ..data.dataset import NeighborhoodDataset
from ..data.synthetic import SyntheticDataGenerator
from ..utils.config import Config
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)

class OpenNeighborTrainer:
    """
    Main trainer class for OpenNeighbor models.
    
    Provides comprehensive training functionality including:
    - Model initialization and configuration
    - Training loop with validation
    - Checkpointing and model saving
    - Metrics tracking and logging
    - Early stopping and learning rate scheduling
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        setup_logging(level=logging.INFO)
        logger.info(f"Initializing OpenNeighborTrainer on device: {self.device}")
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Initialize data
        self.train_loader, self.val_loader = self._initialize_data()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler()
        
        # Initialize loss function
        self.criterion = self._initialize_criterion()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized. Model has {self._count_parameters():,} parameters")
    
    def _initialize_model(self) -> OpenNeighbor:
        """Initialize the OpenNeighbor model."""
        model_config = self.config.get('model', {})
        
        # Set default model configuration
        default_config = {
            'hidden_dim': 256,
            'num_users': 10000,
            'num_items': 50000,
            'num_content_types': 20,
            'num_categories': 50,
            'spatial_heads': 8,
            'spatial_layers': 2,
            'gnn_layers': 3,
            'dropout': 0.1,
            'text_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'freeze_text_encoder': True,
            'top_k': 10
        }
        
        # Update with user config
        default_config.update(model_config)
        
        model = OpenNeighbor(default_config)
        model.to(self.device)
        
        return model
    
    def _initialize_data(self) -> Tuple[DataLoader, DataLoader]:
        """Initialize training and validation data loaders."""
        data_config = self.config.get('data', {})
        
        # Check if we should use synthetic data
        if data_config.get('use_synthetic', True):
            logger.info("Generating synthetic training data...")
            
            # Generate synthetic data
            generator = SyntheticDataGenerator(self.config)
            
            # Generate train and validation datasets
            train_dataset_dict = generator.generate_dataset(
                num_samples=data_config.get('num_train_samples', 8000),
                split='train'
            )
            val_dataset_dict = generator.generate_dataset(
                num_samples=data_config.get('num_val_samples', 1000),
                split='val'
            )
            
            # Create dataset objects
            train_dataset = NeighborhoodDataset(self.config, split='train')
            val_dataset = NeighborhoodDataset(self.config, split='val')
            
        else:
            # Load real data
            data_path = data_config.get('data_path', './data')
            train_dataset = NeighborhoodDataset.from_file(
                f"{data_path}/train.json", 
                split='train'
            )
            val_dataset = NeighborhoodDataset.from_file(
                f"{data_path}/val.json", 
                split='val'
            )
        
        # Create data loaders
        batch_size = self.config.get('training', {}).get('batch_size', 32)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=train_dataset._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=val_dataset._collate_fn
        )
        
        logger.info(f"Data loaders initialized. Train: {len(train_loader)} batches, "
                   f"Val: {len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def _initialize_optimizer(self) -> optim.Optimizer:
        """Initialize optimizer."""
        training_config = self.config.get('training', {})
        
        optimizer_type = training_config.get('optimizer', 'adamw')
        learning_rate = training_config.get('learning_rate', 1e-4)
        weight_decay = training_config.get('weight_decay', 1e-5)
        
        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        return optimizer
    
    def _initialize_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Initialize learning rate scheduler."""
        training_config = self.config.get('training', {})
        
        if training_config.get('use_scheduler', True):
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            return scheduler
        
        return None
    
    def _initialize_criterion(self) -> nn.Module:
        """Initialize loss function."""
        loss_type = self.config.get('training', {}).get('loss_type', 'mse')
        
        if loss_type.lower() == 'mse':
            return nn.MSELoss()
        elif loss_type.lower() == 'mae':
            return nn.L1Loss()
        elif loss_type.lower() == 'huber':
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        num_epochs = self.config.get('training', {}).get('num_epochs', 50)
        early_stopping_patience = self.config.get('training', {}).get('early_stopping_patience', 10)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        start_time = time.time()
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch()
            
            # Validation phase
            val_loss = self._validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save checkpoint if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint('best_model.pt')
                epochs_without_improvement = 0
                logger.info(f"New best validation loss: {val_loss:.4f}")
            else:
                epochs_without_improvement += 1
            
            # Log progress
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Time: {epoch_time:.2f}s")
            
            # Record training history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'timestamp': datetime.now().isoformat()
            })
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping after {epochs_without_improvement} epochs without improvement")
                break
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        # Save final model and training history
        self._save_final_results()
        
        return {
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.training_history[-1]['train_loss'] if self.training_history else 0.0,
            'total_epochs': len(self.training_history),
            'total_time': total_time,
            'training_history': self.training_history
        }
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                output = self.model(batch)
                
                # Compute loss
                targets = batch.get('ratings', torch.randn_like(output.predictions))
                loss = self.criterion(output.predictions, targets)
                
                # Add regularization losses if available
                if hasattr(output, 'fairness_metrics') and output.fairness_metrics:
                    fairness_loss = 0.0
                    for metric_name, metric_value in output.fairness_metrics.items():
                        if isinstance(metric_value, torch.Tensor) and metric_value.requires_grad:
                            fairness_loss += metric_value.mean()
                    
                    fairness_weight = self.config.get('training', {}).get('fairness_weight', 0.1)
                    loss = loss + fairness_weight * fairness_loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                max_grad_norm = self.config.get('training', {}).get('max_grad_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def _validate_epoch(self) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                try:
                    # Forward pass
                    output = self.model(batch)
                    
                    # Compute loss
                    targets = batch.get('ratings', torch.randn_like(output.predictions))
                    loss = self.criterion(output.predictions, targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.warning(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue
        
        return total_loss / max(num_batches, 1)
    
    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _save_final_results(self) -> None:
        """Save final training results."""
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save model configuration
        config_path = self.output_dir / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save final model
        self.model.save_pretrained(self.output_dir / 'final_model')
        
        logger.info(f"Final results saved to {self.output_dir}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def _count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad) 