"""
Basic Usage Examples for OpenNeighbor

This script demonstrates the core functionality of OpenNeighbor
including model initialization, training, and inference.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import numpy as np
import torch
from pathlib import Path

# Import OpenNeighbor components
from openneighbor.core.models.openneighbor import OpenNeighbor
from openneighbor.core.training.trainer import OpenNeighborTrainer
from openneighbor.core.data.synthetic import SyntheticDataGenerator
from openneighbor.core.data.dataset import NeighborhoodDataset


def example_1_basic_model_usage():
    """Example 1: Basic model initialization and recommendation generation."""
    print("=" * 60)
    print("Example 1: Basic Model Usage")
    print("=" * 60)
    
    # Define model configuration
    config = {
        'hidden_dim': 256,
        'num_users': 1000,
        'num_items': 5000,
        'spatial_heads': 8,
        'gnn_layers': 3,
        'dropout': 0.1
    }
    
    # Initialize model
    print("Initializing OpenNeighbor model...")
    model = OpenNeighbor(config)
    print(f"Model created with {model.count_parameters():,} parameters")
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    user_id = 42
    candidate_items = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    context = {'coordinates': [37.7749, -122.4194]}  # San Francisco
    
    recommendations = model.recommend(
        user_id=user_id,
        candidate_items=candidate_items,
        context=context,
        top_k=5
    )
    
    print(f"\nTop 5 recommendations for User {user_id}:")
    for i, (item_id, score) in enumerate(recommendations, 1):
        print(f"  {i}. Item {item_id}: Score {score:.4f}")
    
    # Get explanation
    print("\nGetting explanation for top recommendation...")
    explanation = model.explain_recommendation(
        user_id=user_id,
        item_id=recommendations[0][0],  # First element is item_id
        context=context
    )
    
    print(f"Explanation factors: {len(explanation['explanation_factors'])}")
    for factor in explanation['explanation_factors'][:3]:
        print(f"  - {factor['name']}: {factor['contribution']:.4f}")


def example_2_synthetic_data_generation():
    """Example 2: Synthetic data generation for development."""
    print("\n" + "=" * 60)
    print("Example 2: Synthetic Data Generation")
    print("=" * 60)
    
    # Configure synthetic data generator
    config = {
        'num_users': 500,
        'num_items': 2000,
        'num_neighborhoods': 20,
        'spatial_radius_km': 3.0,
        'random_seed': 42
    }
    
    # Initialize generator
    print("Initializing synthetic data generator...")
    generator = SyntheticDataGenerator(config)
    
    # Generate training dataset
    print("Generating training dataset...")
    train_data = generator.generate_dataset(
        num_samples=5000,
        split='train'
    )
    
    print(f"Generated dataset with:")
    print(f"  - {len(train_data['users'])} users")
    print(f"  - {len(train_data['items'])} items")
    print(f"  - {len(train_data['interactions'])} interactions")
    print(f"  - {len(train_data['neighborhoods'])} neighborhoods")
    
    # Show sample data
    print("\nSample user:")
    sample_user = train_data['users'][0]
    print(f"  User {sample_user['user_id']}: {sample_user['age_group']}")
    print(f"  Location: ({sample_user['latitude']:.4f}, {sample_user['longitude']:.4f})")
    print(f"  Neighborhood: {sample_user['neighborhood_id']}")
    
    print("\nSample item:")
    sample_item = train_data['items'][0]
    print(f"  Item {sample_item['item_id']}: {sample_item['name']}")
    print(f"  Category: {sample_item['category']}")
    print(f"  Location: ({sample_item['latitude']:.4f}, {sample_item['longitude']:.4f})")
    print(f"  Rating: {sample_item['rating']:.2f}")


def example_3_model_training():
    """Example 3: Training a model with synthetic data."""
    print("\n" + "=" * 60)
    print("Example 3: Model Training")
    print("=" * 60)
    
    # Define training configuration
    config = {
        'model': {
            'hidden_dim': 128,
            'num_users': 200,
            'num_items': 1000,
            'spatial_heads': 4,
            'gnn_layers': 2,
            'dropout': 0.1
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 3,  # Small number for demo
            'early_stopping_patience': 5
        },
        'data': {
            'use_synthetic': True,
            'num_train_samples': 1000,
            'num_val_samples': 200
        },
        'output_dir': './example_outputs'
    }
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = OpenNeighborTrainer(config)
    
    # Start training
    print("Starting training...")
    results = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Final training loss: {results['final_train_loss']:.4f}")
    print(f"Final validation loss: {results['final_val_loss']:.4f}")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Training time: {results['training_time']:.2f} seconds")


def example_4_batch_recommendations():
    """Example 4: Batch recommendation generation."""
    print("\n" + "=" * 60)
    print("Example 4: Batch Recommendations")
    print("=" * 60)
    
    # Initialize model
    config = {
        'hidden_dim': 128,
        'num_users': 500,
        'num_items': 2000,
        'spatial_heads': 4,
        'gnn_layers': 2
    }
    
    model = OpenNeighbor(config)
    print(f"Model initialized with {model.count_parameters():,} parameters")
    
    # Prepare batch data
    user_ids = [10, 25, 50, 75, 100]
    candidate_items = [
        list(range(100, 110)),
        list(range(200, 210)),
        list(range(300, 310)),
        list(range(400, 410)),
        list(range(500, 510))
    ]
    contexts = [
        {'coordinates': [37.7749, -122.4194]},  # San Francisco
        {'coordinates': [40.7128, -74.0060]},   # New York
        {'coordinates': [34.0522, -118.2437]},  # Los Angeles
        {'coordinates': [41.8781, -87.6298]},   # Chicago
        {'coordinates': [29.7604, -95.3698]}    # Houston
    ]
    
    # Generate batch recommendations
    print("Generating batch recommendations...")
    batch_recommendations = model.recommend_batch(
        user_ids=user_ids,
        candidate_items=candidate_items,
        contexts=contexts,
        top_k=3
    )
    
    # Display results
    print(f"\nBatch recommendations for {len(user_ids)} users:")
    for i, (user_id, recs) in enumerate(zip(user_ids, batch_recommendations)):
        city_names = ['San Francisco', 'New York', 'Los Angeles', 'Chicago', 'Houston']
        print(f"\nUser {user_id} in {city_names[i]}:")
        for j, (item_id, score) in enumerate(recs, 1):
            print(f"  {j}. Item {item_id}: Score {score:.4f}")


def example_5_spatial_analysis():
    """Example 5: Spatial analysis and neighborhood effects."""
    print("\n" + "=" * 60)
    print("Example 5: Spatial Analysis")
    print("=" * 60)
    
    # Initialize model
    config = {
        'hidden_dim': 128,
        'num_users': 300,
        'num_items': 1500,
        'spatial_heads': 8,
        'gnn_layers': 3
    }
    
    model = OpenNeighbor(config)
    print("Analyzing spatial effects on recommendations...")
    
    # Test same user in different locations
    user_id = 42
    candidate_items = list(range(100, 120))
    
    locations = [
        {'name': 'Downtown', 'coordinates': [37.7749, -122.4194]},
        {'name': 'Suburbs', 'coordinates': [37.7849, -122.4094]},
        {'name': 'Waterfront', 'coordinates': [37.7649, -122.4294]},
        {'name': 'Hills', 'coordinates': [37.7949, -122.4394]}
    ]
    
    print(f"\nRecommendations for User {user_id} in different locations:")
    
    for location in locations:
        recommendations = model.recommend(
            user_id=user_id,
            candidate_items=candidate_items,
            context=location,
            top_k=3
        )
        
        print(f"\n{location['name']} ({location['coordinates'][0]:.4f}, {location['coordinates'][1]:.4f}):")
        for i, (item_id, score) in enumerate(recommendations, 1):
            print(f"  {i}. Item {item_id}: Score {score:.4f}")


def main():
    """Run all examples."""
    print("OpenNeighbor Usage Examples")
    print("Author: Nik Jois <nikjois@llamasearch.ai>")
    print("Version: 1.0.0")
    
    try:
        # Run all examples
        example_1_basic_model_usage()
        example_2_synthetic_data_generation()
        example_3_model_training()
        example_4_batch_recommendations()
        example_5_spatial_analysis()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 