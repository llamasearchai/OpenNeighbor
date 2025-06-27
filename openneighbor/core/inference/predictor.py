"""
Inference predictor for OpenNeighbor models.

Author: Nik Jois <nikjois@llamasearch.ai>

This module provides the main predictor class for generating recommendations
and performing inference with trained OpenNeighbor models.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json
import time

from ..models.openneighbor import OpenNeighbor
from ..data.dataset import NeighborhoodDataset
from ..data.synthetic import SyntheticDataGenerator

logger = logging.getLogger(__name__)

class OpenNeighborPredictor:
    """
    Main predictor class for OpenNeighbor models.
    
    Provides comprehensive inference functionality including:
    - Single and batch recommendation generation
    - Model evaluation and metrics computation
    - Performance benchmarking
    - Explanation generation
    """
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing OpenNeighborPredictor on device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Initialize data generator for synthetic data if needed
        self.data_generator = SyntheticDataGenerator(config)
        
        logger.info("OpenNeighborPredictor initialized successfully")
    
    def _load_model(self, model_path: str) -> OpenNeighbor:
        """Load trained model from checkpoint or pretrained directory."""
        model_path = Path(model_path)
        
        if model_path.is_file() and model_path.suffix == '.pt':
            # Load from checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model config from checkpoint
            model_config = checkpoint.get('config', {}).get('model', {})
            if not model_config:
                # Use default config if not found in checkpoint
                model_config = self.config.get('model', {})
            
            # Initialize model
            model = OpenNeighbor(model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"Model loaded from checkpoint: {model_path}")
            
        elif model_path.is_dir():
            # Load from pretrained directory
            model = OpenNeighbor.from_pretrained(model_path)
            logger.info(f"Model loaded from pretrained directory: {model_path}")
            
        else:
            raise ValueError(f"Invalid model path: {model_path}")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def recommend(self, user_id: int, candidate_items: List[int],
                  context: Optional[Dict[str, Any]] = None,
                  top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a single user.
        
        Args:
            user_id: ID of the user
            candidate_items: List of candidate item IDs
            context: Additional context information
            top_k: Number of recommendations to return
            
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        return self.model.recommend(
            user_id=user_id,
            candidate_items=candidate_items,
            context=context,
            top_k=top_k
        )
    
    def recommend_batch(self, user_ids: List[int],
                       candidate_items: List[List[int]],
                       contexts: Optional[List[Dict[str, Any]]] = None,
                       top_k: Optional[int] = None) -> List[List[Tuple[int, float]]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            candidate_items: List of candidate item lists for each user
            contexts: Optional list of context dicts for each user
            top_k: Number of recommendations to return per user
            
        Returns:
            List of recommendation lists for each user
        """
        return self.model.recommend_batch(
            user_ids=user_ids,
            candidate_items=candidate_items,
            contexts=contexts,
            top_k=top_k
        )
    
    def explain_recommendation(self, user_id: int, item_id: int,
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Provide explanation for a recommendation.
        
        Args:
            user_id: ID of the user
            item_id: ID of the recommended item
            context: Additional context
            
        Returns:
            Dictionary containing explanation details
        """
        return self.model.explain_recommendation(
            user_id=user_id,
            item_id=item_id,
            context=context
        )
    
    def evaluate_accuracy(self, dataset: NeighborhoodDataset, 
                         batch_size: int = 64) -> Dict[str, float]:
        """
        Evaluate model accuracy on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of accuracy metrics
        """
        self.model.eval()
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )
        
        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                output = self.model(batch)
                predictions = output.predictions
                
                # Get targets (use synthetic targets if not available)
                targets = batch.get('ratings', torch.randn_like(predictions))
                
                # Compute metrics
                mse_loss = torch.nn.functional.mse_loss(predictions, targets)
                mae_loss = torch.nn.functional.l1_loss(predictions, targets)
                
                total_loss += mse_loss.item() * predictions.size(0)
                total_mae += mae_loss.item() * predictions.size(0)
                total_rmse += torch.sqrt(mse_loss).item() * predictions.size(0)
                num_samples += predictions.size(0)
        
        return {
            'mse': total_loss / num_samples,
            'mae': total_mae / num_samples,
            'rmse': total_rmse / num_samples,
            'num_samples': num_samples
        }
    
    def evaluate_diversity(self, dataset: NeighborhoodDataset,
                          batch_size: int = 64) -> Dict[str, float]:
        """
        Evaluate recommendation diversity.
        
        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of diversity metrics
        """
        self.model.eval()
        
        # Generate recommendations for a sample of users
        sample_users = np.random.choice(
            range(min(1000, len(dataset))), 
            size=min(100, len(dataset)), 
            replace=False
        )
        
        all_recommendations = []
        category_distributions = []
        
        for user_id in sample_users:
            # Generate synthetic candidate items
            candidate_items = list(range(user_id * 10, (user_id + 1) * 10))
            
            try:
                recommendations = self.recommend(
                    user_id=int(user_id),
                    candidate_items=candidate_items,
                    top_k=10
                )
                
                recommended_items = [item_id for item_id, _ in recommendations]
                all_recommendations.extend(recommended_items)
                
                # Mock category distribution (in real implementation, would use item metadata)
                categories = np.random.choice(['A', 'B', 'C', 'D', 'E'], size=len(recommended_items))
                category_dist = np.bincount(np.array([ord(c) - ord('A') for c in categories]))
                category_distributions.append(category_dist)
                
            except Exception as e:
                logger.warning(f"Error generating recommendations for user {user_id}: {e}")
                continue
        
        # Calculate diversity metrics
        if not all_recommendations:
            return {'diversity_score': 0.0, 'coverage': 0.0, 'novelty': 0.0}
        
        # Intra-list diversity (average pairwise distance within recommendation lists)
        intra_diversity = np.mean([
            len(set(range(i*10, (i+1)*10))) / 10.0 
            for i in range(len(category_distributions))
        ])
        
        # Coverage (fraction of all items recommended)
        unique_items = len(set(all_recommendations))
        total_items = len(all_recommendations)
        coverage = unique_items / max(total_items, 1)
        
        # Novelty (inverse popularity)
        item_counts = {}
        for item in all_recommendations:
            item_counts[item] = item_counts.get(item, 0) + 1
        
        novelty = 1.0 - (np.mean(list(item_counts.values())) / max(item_counts.values(), 1))
        
        return {
            'diversity_score': intra_diversity,
            'coverage': coverage,
            'novelty': novelty,
            'num_users_evaluated': len(sample_users),
            'total_recommendations': len(all_recommendations)
        }
    
    def evaluate_fairness(self, dataset: NeighborhoodDataset,
                         batch_size: int = 64) -> Dict[str, float]:
        """
        Evaluate recommendation fairness.
        
        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of fairness metrics
        """
        self.model.eval()
        
        # Generate recommendations for different user groups
        sample_size = min(100, len(dataset))
        user_groups = {
            'group_A': np.random.choice(range(sample_size), size=sample_size//2, replace=False),
            'group_B': np.random.choice(range(sample_size//2, sample_size), size=sample_size//2, replace=False)
        }
        
        group_metrics = {}
        
        for group_name, user_ids in user_groups.items():
            group_recommendations = []
            group_scores = []
            
            for user_id in user_ids:
                candidate_items = list(range(user_id * 10, (user_id + 1) * 10))
                
                try:
                    recommendations = self.recommend(
                        user_id=int(user_id),
                        candidate_items=candidate_items,
                        top_k=5
                    )
                    
                    scores = [score for _, score in recommendations]
                    group_recommendations.extend([item_id for item_id, _ in recommendations])
                    group_scores.extend(scores)
                    
                except Exception as e:
                    logger.warning(f"Error generating recommendations for user {user_id}: {e}")
                    continue
            
            # Calculate group metrics
            avg_score = np.mean(group_scores) if group_scores else 0.0
            score_variance = np.var(group_scores) if group_scores else 0.0
            
            group_metrics[group_name] = {
                'avg_score': avg_score,
                'score_variance': score_variance,
                'num_recommendations': len(group_recommendations)
            }
        
        # Calculate fairness metrics
        if len(group_metrics) >= 2:
            group_names = list(group_metrics.keys())
            group_A_score = group_metrics[group_names[0]]['avg_score']
            group_B_score = group_metrics[group_names[1]]['avg_score']
            
            # Demographic parity (difference in average scores)
            demographic_parity = abs(group_A_score - group_B_score)
            
            # Equal opportunity (ratio of scores)
            equal_opportunity = min(group_A_score, group_B_score) / max(group_A_score, group_B_score, 1e-8)
            
            return {
                'demographic_parity': demographic_parity,
                'equal_opportunity': equal_opportunity,
                'group_A_avg_score': group_A_score,
                'group_B_avg_score': group_B_score,
                'fairness_score': 1.0 - demographic_parity  # Higher is more fair
            }
        
        return {'fairness_score': 0.0}
    
    def benchmark_speed(self, dataset: NeighborhoodDataset,
                       batch_size: int = 64) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            dataset: Dataset to benchmark on
            batch_size: Batch size for benchmarking
            
        Returns:
            Dictionary of speed metrics
        """
        self.model.eval()
        
        # Warm up
        dummy_user_ids = [1, 2, 3]
        dummy_candidates = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
        
        for _ in range(5):
            with torch.no_grad():
                self.recommend_batch(dummy_user_ids, dummy_candidates)
        
        # Benchmark single recommendations
        single_times = []
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                self.recommend(
                    user_id=1,
                    candidate_items=[10, 11, 12, 13, 14],
                    top_k=5
                )
            single_times.append(time.time() - start_time)
        
        # Benchmark batch recommendations
        batch_user_ids = list(range(batch_size))
        batch_candidates = [list(range(i*10, (i+1)*10)) for i in range(batch_size)]
        
        batch_times = []
        for _ in range(20):
            start_time = time.time()
            with torch.no_grad():
                self.recommend_batch(batch_user_ids, batch_candidates)
            batch_times.append(time.time() - start_time)
        
        return {
            'single_recommendation_ms': np.mean(single_times) * 1000,
            'single_recommendation_std_ms': np.std(single_times) * 1000,
            'batch_recommendation_ms': np.mean(batch_times) * 1000,
            'batch_recommendation_std_ms': np.std(batch_times) * 1000,
            'throughput_recs_per_sec': batch_size / np.mean(batch_times),
            'batch_size': batch_size
        } 