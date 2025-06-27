"""
Performance Benchmarks for OpenNeighbor

Comprehensive benchmarking suite to measure inference speed, throughput,
and memory usage of the OpenNeighbor recommendation system.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Any
from openneighbor.core.models.openneighbor import OpenNeighbor
from openneighbor.core.inference.predictor import OpenNeighborPredictor
from openneighbor.core.data.synthetic import SyntheticDataGenerator


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for OpenNeighbor."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the benchmark with model configuration."""
        self.config = config
        self.model = OpenNeighbor(config)
        self.predictor = OpenNeighborPredictor(self.model)
        self.results = {}
    
    def benchmark_inference_speed(self, num_requests: int = 1000) -> Dict[str, float]:
        """Benchmark single recommendation inference speed."""
        print(f"Benchmarking inference speed with {num_requests} requests...")
        
        # Prepare test data
        user_ids = np.random.randint(0, self.config['num_users'], num_requests)
        candidate_items = [
            np.random.choice(self.config['num_items'], size=10, replace=False).tolist()
            for _ in range(num_requests)
        ]
        contexts = [
            {'coordinates': [np.random.uniform(37.7, 37.8), np.random.uniform(-122.5, -122.4)]}
            for _ in range(num_requests)
        ]
        
        # Warm up
        for i in range(10):
            self.model.recommend(
                user_id=user_ids[i],
                candidate_items=candidate_items[i],
                context=contexts[i],
                top_k=5
            )
        
        # Benchmark
        start_time = time.time()
        for i in range(num_requests):
            self.model.recommend(
                user_id=user_ids[i],
                candidate_items=candidate_items[i],
                context=contexts[i],
                top_k=5
            )
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_requests
        throughput = num_requests / total_time
        
        results = {
            'total_time_seconds': total_time,
            'average_time_ms': avg_time * 1000,
            'throughput_rps': throughput,
            'num_requests': num_requests
        }
        
        self.results['inference_speed'] = results
        return results
    
    def benchmark_batch_processing(self, batch_sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark batch processing performance."""
        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32, 64, 128]
        
        print(f"Benchmarking batch processing with sizes: {batch_sizes}")
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            # Prepare batch data
            requests = []
            for _ in range(batch_size):
                requests.append({
                    'user_id': np.random.randint(0, self.config['num_users']),
                    'candidate_items': np.random.choice(
                        self.config['num_items'], size=10, replace=False
                    ).tolist(),
                    'context': {
                        'coordinates': [
                            np.random.uniform(37.7, 37.8),
                            np.random.uniform(-122.5, -122.4)
                        ]
                    }
                })
            
            # Warm up
            self.predictor.predict_batch(requests[:min(5, batch_size)])
            
            # Benchmark
            start_time = time.time()
            results = self.predictor.predict_batch(requests)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_request = total_time / batch_size
            throughput = batch_size / total_time
            
            batch_results[batch_size] = {
                'total_time_seconds': total_time,
                'avg_time_per_request_ms': avg_time_per_request * 1000,
                'throughput_rps': throughput,
                'batch_size': batch_size
            }
        
        self.results['batch_processing'] = batch_results
        return batch_results
    
    def benchmark_memory_usage(self) -> Dict[str, float]:
        """Benchmark memory usage during inference."""
        print("Benchmarking memory usage...")
        
        process = psutil.Process()
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load model and perform inference
        user_id = np.random.randint(0, self.config['num_users'])
        candidate_items = np.random.choice(
            self.config['num_items'], size=100, replace=False
        ).tolist()
        context = {'coordinates': [37.7749, -122.4194]}
        
        # Perform multiple inferences to measure peak memory
        for _ in range(100):
            self.model.recommend(
                user_id=user_id,
                candidate_items=candidate_items,
                context=context,
                top_k=10
            )
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        results = {
            'baseline_memory_mb': baseline_memory,
            'peak_memory_mb': peak_memory,
            'memory_increase_mb': memory_increase,
            'model_parameters': self.model.count_parameters()
        }
        
        self.results['memory_usage'] = results
        return results
    
    def benchmark_scalability(self, user_counts: List[int] = None) -> Dict[str, Any]:
        """Benchmark scalability with different user counts."""
        if user_counts is None:
            user_counts = [1000, 5000, 10000, 50000, 100000]
        
        print(f"Benchmarking scalability with user counts: {user_counts}")
        
        scalability_results = {}
        
        for num_users in user_counts:
            # Create model with different user count
            config = self.config.copy()
            config['num_users'] = num_users
            
            model = OpenNeighbor(config)
            
            # Benchmark initialization time
            start_time = time.time()
            model = OpenNeighbor(config)
            init_time = time.time() - start_time
            
            # Benchmark inference time
            user_id = np.random.randint(0, num_users)
            candidate_items = np.random.choice(
                self.config['num_items'], size=10, replace=False
            ).tolist()
            context = {'coordinates': [37.7749, -122.4194]}
            
            start_time = time.time()
            for _ in range(10):
                model.recommend(
                    user_id=user_id,
                    candidate_items=candidate_items,
                    context=context,
                    top_k=5
                )
            inference_time = (time.time() - start_time) / 10
            
            scalability_results[num_users] = {
                'num_users': num_users,
                'init_time_seconds': init_time,
                'avg_inference_time_ms': inference_time * 1000,
                'model_parameters': model.count_parameters()
            }
        
        self.results['scalability'] = scalability_results
        return scalability_results
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks and return comprehensive results."""
        print("Running comprehensive OpenNeighbor benchmarks...")
        print("=" * 60)
        
        # Run all benchmarks
        self.benchmark_inference_speed()
        self.benchmark_batch_processing()
        self.benchmark_memory_usage()
        self.benchmark_scalability()
        
        # Add system information
        self.results['system_info'] = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'python_version': torch.__version__,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        return self.results
    
    def print_results(self):
        """Print formatted benchmark results."""
        if not self.results:
            print("No benchmark results available. Run benchmarks first.")
            return
        
        print("\nOpenNeighbor Performance Benchmark Results")
        print("=" * 60)
        
        # System info
        if 'system_info' in self.results:
            info = self.results['system_info']
            print(f"\nSystem Information:")
            print(f"  CPU Cores: {info['cpu_count']}")
            print(f"  Memory: {info['memory_total_gb']:.1f} GB")
            print(f"  PyTorch: {info['torch_version']}")
            print(f"  Device: {info['device']}")
        
        # Inference speed
        if 'inference_speed' in self.results:
            speed = self.results['inference_speed']
            print(f"\nInference Speed:")
            print(f"  Average Time: {speed['average_time_ms']:.2f} ms")
            print(f"  Throughput: {speed['throughput_rps']:.1f} requests/second")
            print(f"  Total Requests: {speed['num_requests']}")
        
        # Memory usage
        if 'memory_usage' in self.results:
            memory = self.results['memory_usage']
            print(f"\nMemory Usage:")
            print(f"  Baseline: {memory['baseline_memory_mb']:.1f} MB")
            print(f"  Peak: {memory['peak_memory_mb']:.1f} MB")
            print(f"  Increase: {memory['memory_increase_mb']:.1f} MB")
            print(f"  Model Parameters: {memory['model_parameters']:,}")
        
        # Batch processing
        if 'batch_processing' in self.results:
            print(f"\nBatch Processing Performance:")
            for batch_size, result in self.results['batch_processing'].items():
                print(f"  Batch Size {batch_size:3d}: {result['throughput_rps']:.1f} RPS, "
                      f"{result['avg_time_per_request_ms']:.2f} ms/request")
        
        print("\n" + "=" * 60)


def main():
    """Run performance benchmarks with default configuration."""
    config = {
        'hidden_dim': 256,
        'num_users': 10000,
        'num_items': 50000,
        'spatial_heads': 8,
        'gnn_layers': 3,
        'dropout': 0.1
    }
    
    benchmark = PerformanceBenchmark(config)
    results = benchmark.run_all_benchmarks()
    benchmark.print_results()
    
    return results


if __name__ == '__main__':
    main() 