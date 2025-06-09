import pytest
import torch
from sam2.modeling.memory_strategies.strategies.lru_quantized import LRUQuantizedMemory

def test_lru_quantized_initialization():
    """Test initialization of LRU Quantized Memory"""
    memory = LRUQuantizedMemory(
        max_memory_size=3,
        usage_threshold=2,
        removal_threshold=1
    )
    assert memory.max_memory_size == 3
    assert memory.usage_threshold == 2
    assert memory.removal_threshold == 1
    assert memory.memory_size == 0
    assert not memory.is_full

def test_add_memory():
    """Test adding memories to the bank"""
    memory = LRUQuantizedMemory(max_memory_size=2)
    
    # Add first memory
    tensor1 = torch.randn(3, 3)
    memory.add_memory(0, {"features": tensor1})
    assert memory.memory_size == 1
    
    # Add second memory
    tensor2 = torch.randn(3, 3)
    memory.add_memory(1, {"features": tensor2})
    assert memory.memory_size == 2
    
    # Add third memory (should remove least used)
    tensor3 = torch.randn(3, 3)
    memory.add_memory(2, {"features": tensor3})
    assert memory.memory_size == 2

def test_get_memory():
    """Test retrieving memories with quantization"""
    memory = LRUQuantizedMemory(
        max_memory_size=3,
        usage_threshold=2,
        removal_threshold=1
    )
    
    # Add memory
    tensor = torch.randn(3, 3)
    memory.add_memory(0, {"features": tensor})
    
    # Get memory first time (should be 8-bit)
    mem1 = memory.get_memory(0)
    assert mem1 is not None
    assert mem1["features"].dtype == torch.float32  # First access is full precision
    
    # Get memory second time (should be 16-bit)
    mem2 = memory.get_memory(0)
    assert mem2["features"].dtype == torch.float16
    
    # Get memory third time (should be 32-bit)
    mem3 = memory.get_memory(0)
    assert mem3["features"].dtype == torch.float32

def test_remove_least_used():
    """Test removal of least used memories"""
    memory = LRUQuantizedMemory(
        max_memory_size=2,
        usage_threshold=2,
        removal_threshold=1
    )
    
    # Add two memories
    tensor1 = torch.randn(3, 3)
    tensor2 = torch.randn(3, 3)
    memory.add_memory(0, {"features": tensor1})
    memory.add_memory(1, {"features": tensor2})
    
    # Use first memory more
    memory.get_memory(0)
    memory.get_memory(0)
    
    # Add third memory (should remove second memory as it's least used)
    tensor3 = torch.randn(3, 3)
    memory.add_memory(2, {"features": tensor3})
    
    assert memory.memory_size == 2
    assert 0 in memory._memory_bank  # First memory should remain
    assert 2 in memory._memory_bank  # New memory should be added
    assert 1 not in memory._memory_bank  # Second memory should be removed

def test_get_memory_stats():
    """Test memory statistics"""
    memory = LRUQuantizedMemory(max_memory_size=3)
    
    # Add some memories
    tensor1 = torch.randn(3, 3)
    tensor2 = torch.randn(3, 3)
    memory.add_memory(0, {"features": tensor1})
    memory.add_memory(1, {"features": tensor2})
    
    # Use memories
    memory.get_memory(0)
    memory.get_memory(0)
    memory.get_memory(1)
    
    stats = memory.get_memory_stats()
    assert stats["total_memories"] == 2
    assert stats["max_memory_size"] == 3
    assert stats["usage_counts"][0] == 2
    assert stats["usage_counts"][1] == 1
    assert len(stats["memory_indices"]) == 2
    assert stats["average_usage"] == 1.5

def test_clear():
    """Test clearing the memory bank"""
    memory = LRUQuantizedMemory(max_memory_size=3)
    
    # Add some memories
    tensor1 = torch.randn(3, 3)
    tensor2 = torch.randn(3, 3)
    memory.add_memory(0, {"features": tensor1})
    memory.add_memory(1, {"features": tensor2})
    
    # Clear memory bank
    memory.clear()
    
    assert memory.memory_size == 0
    assert not memory._memory_bank
    assert not memory._usage_count 