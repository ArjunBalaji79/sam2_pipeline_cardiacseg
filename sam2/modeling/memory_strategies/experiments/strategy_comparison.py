"""
Compare different configurations of memory strategies.
"""
import torch
import time
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np

from ..strategies.lru_quantized import LRUQuantizedMemory

class MemoryStrategyComparison:
    def __init__(self):
        self.strategies = {
            "aggressive": LRUQuantizedMemory(
                max_memory_size=3,
                usage_threshold=2,
                removal_threshold=1
            ),
            "conservative": LRUQuantizedMemory(
                max_memory_size=3,
                usage_threshold=3,
                removal_threshold=2
            ),
            "balanced": LRUQuantizedMemory(
                max_memory_size=3,
                usage_threshold=2,
                removal_threshold=2
            )
        }
        
    def create_sample_memory(self, size: int = 32) -> Dict[str, Any]:
        """Create a sample memory tensor."""
        return {
            "features": torch.randn(3, size, size),
            "mask": torch.randn(1, size, size),
            "metadata": {"timestamp": time.time()}
        }
    
    def run_memory_operations(self, num_operations: int = 100) -> Dict[str, Dict[str, Any]]:
        """Run a series of memory operations and collect statistics."""
        results = {name: {
            "memory_usage": [],
            "operation_times": [],
            "quantization_levels": []
        } for name in self.strategies}
        
        for i in range(num_operations):
            # Add memory
            for name, strategy in self.strategies.items():
                start_time = time.time()
                strategy.add_memory(i, self.create_sample_memory())
                results[name]["operation_times"].append(time.time() - start_time)
                
                # Randomly access some memories
                if i > 0:
                    access_idx = np.random.randint(0, i)
                    start_time = time.time()
                    memory = strategy.get_memory(access_idx)
                    results[name]["operation_times"].append(time.time() - start_time)
                    
                    if memory is not None:
                        results[name]["quantization_levels"].append(
                            memory["features"].dtype
                        )
                
                # Get memory stats
                stats = strategy.get_memory_stats()
                results[name]["memory_usage"].append(stats["total_memories"])
        
        return results
    
    def plot_results(self, results: Dict[str, Dict[str, Any]]):
        """Plot comparison results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot memory usage over time
        for name, data in results.items():
            ax1.plot(data["memory_usage"], label=name)
        ax1.set_title("Memory Usage Over Time")
        ax1.set_xlabel("Operation")
        ax1.set_ylabel("Number of Memories")
        ax1.legend()
        
        # Plot operation times
        for name, data in results.items():
            ax2.plot(data["operation_times"], label=name, alpha=0.7)
        ax2.set_title("Operation Times")
        ax2.set_xlabel("Operation")
        ax2.set_ylabel("Time (seconds)")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("memory_strategy_comparison.png")
        plt.close()
    
    def print_statistics(self, results: Dict[str, Dict[str, Any]]):
        """Print detailed statistics."""
        print("\nMemory Strategy Comparison Results")
        print("=" * 50)
        
        for name, data in results.items():
            print(f"\n{name.upper()} Strategy:")
            print(f"Average Memory Usage: {np.mean(data['memory_usage']):.2f}")
            print(f"Average Operation Time: {np.mean(data['operation_times'])*1000:.2f}ms")
            
            if data["quantization_levels"]:
                dtype_counts = {}
                for dtype in data["quantization_levels"]:
                    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                print("Quantization Distribution:")
                for dtype, count in dtype_counts.items():
                    print(f"  {dtype}: {count} times")

def main():
    # Create comparison instance
    comparison = MemoryStrategyComparison()
    
    # Run comparison
    print("Running memory strategy comparison...")
    results = comparison.run_memory_operations(num_operations=100)
    
    # Plot results
    print("Generating plots...")
    comparison.plot_results(results)
    
    # Print statistics
    comparison.print_statistics(results)
    
    print("\nResults saved to 'memory_strategy_comparison.png'")

if __name__ == "__main__":
    main() 