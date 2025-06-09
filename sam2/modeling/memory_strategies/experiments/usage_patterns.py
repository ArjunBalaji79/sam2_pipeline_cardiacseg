"""
Demonstrate different memory usage patterns and their effects.
"""
import torch
import time
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

from ..strategies.lru_quantized import LRUQuantizedMemory

class MemoryUsagePatterns:
    def __init__(self, max_memory_size: int = 5):
        self.memory = LRUQuantizedMemory(
            max_memory_size=max_memory_size,
            usage_threshold=2,
            removal_threshold=1
        )
        
    def create_sample_memory(self, size: int = 32) -> Dict[str, Any]:
        """Create a sample memory tensor."""
        return {
            "features": torch.randn(3, size, size),
            "mask": torch.randn(1, size, size),
            "metadata": {"timestamp": time.time()}
        }
    
    def sequential_access(self, num_frames: int = 20) -> Dict[str, List[Any]]:
        """Simulate sequential frame access pattern."""
        results = {
            "memory_usage": [],
            "quantization_levels": [],
            "access_times": []
        }
        
        for i in range(num_frames):
            # Add new memory
            self.memory.add_memory(i, self.create_sample_memory())
            
            # Access previous frames sequentially
            for j in range(max(0, i-2), i+1):
                start_time = time.time()
                memory = self.memory.get_memory(j)
                results["access_times"].append(time.time() - start_time)
                
                if memory is not None:
                    results["quantization_levels"].append(
                        memory["features"].dtype
                    )
            
            results["memory_usage"].append(self.memory.memory_size)
        
        return results
    
    def random_access(self, num_frames: int = 20) -> Dict[str, List[Any]]:
        """Simulate random frame access pattern."""
        results = {
            "memory_usage": [],
            "quantization_levels": [],
            "access_times": []
        }
        
        for i in range(num_frames):
            # Add new memory
            self.memory.add_memory(i, self.create_sample_memory())
            
            # Randomly access some previous frames
            num_accesses = np.random.randint(1, 4)
            for _ in range(num_accesses):
                access_idx = np.random.randint(0, i+1)
                start_time = time.time()
                memory = self.memory.get_memory(access_idx)
                results["access_times"].append(time.time() - start_time)
                
                if memory is not None:
                    results["quantization_levels"].append(
                        memory["features"].dtype
                    )
            
            results["memory_usage"].append(self.memory.memory_size)
        
        return results
    
    def burst_access(self, num_frames: int = 20) -> Dict[str, List[Any]]:
        """Simulate burst access pattern (frequent access to recent frames)."""
        results = {
            "memory_usage": [],
            "quantization_levels": [],
            "access_times": []
        }
        
        for i in range(num_frames):
            # Add new memory
            self.memory.add_memory(i, self.create_sample_memory())
            
            # Burst access to recent frames
            if i > 0:
                for _ in range(3):  # Multiple accesses to recent frames
                    for j in range(max(0, i-2), i+1):
                        start_time = time.time()
                        memory = self.memory.get_memory(j)
                        results["access_times"].append(time.time() - start_time)
                        
                        if memory is not None:
                            results["quantization_levels"].append(
                                memory["features"].dtype
                            )
            
            results["memory_usage"].append(self.memory.memory_size)
        
        return results
    
    def plot_patterns(self, patterns: Dict[str, Dict[str, List[Any]]]):
        """Plot different access patterns."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot memory usage
        for name, data in patterns.items():
            ax1.plot(data["memory_usage"], label=name)
        ax1.set_title("Memory Usage for Different Access Patterns")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Number of Memories")
        ax1.legend()
        
        # Plot access times
        for name, data in patterns.items():
            ax2.plot(data["access_times"], label=name, alpha=0.7)
        ax2.set_title("Access Times for Different Patterns")
        ax2.set_xlabel("Access")
        ax2.set_ylabel("Time (seconds)")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("memory_usage_patterns.png")
        plt.close()
    
    def print_pattern_statistics(self, patterns: Dict[str, Dict[str, List[Any]]]):
        """Print statistics for each access pattern."""
        print("\nMemory Usage Pattern Statistics")
        print("=" * 50)
        
        for name, data in patterns.items():
            print(f"\n{name.upper()} Pattern:")
            print(f"Average Memory Usage: {np.mean(data['memory_usage']):.2f}")
            print(f"Average Access Time: {np.mean(data['access_times'])*1000:.2f}ms")
            
            if data["quantization_levels"]:
                dtype_counts = {}
                for dtype in data["quantization_levels"]:
                    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                print("Quantization Distribution:")
                for dtype, count in dtype_counts.items():
                    print(f"  {dtype}: {count} times")

def main():
    # Create patterns instance
    patterns = MemoryUsagePatterns()
    
    # Run different access patterns
    print("Running memory usage pattern simulations...")
    results = {
        "Sequential": patterns.sequential_access(),
        "Random": patterns.random_access(),
        "Burst": patterns.burst_access()
    }
    
    # Plot results
    print("Generating plots...")
    patterns.plot_patterns(results)
    
    # Print statistics
    patterns.print_pattern_statistics(results)
    
    print("\nResults saved to 'memory_usage_patterns.png'")

if __name__ == "__main__":
    main() 