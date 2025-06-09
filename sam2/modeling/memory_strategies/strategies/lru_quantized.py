from collections import OrderedDict
from typing import Dict, List, Optional, Any
import torch

from ..base import BaseMemoryStrategy
from ..utils.quantization import quantize_tensor, get_quantization_level

class LRUQuantizedMemory(BaseMemoryStrategy):
    """
    LRU (Least Recently Used) Memory Bank with Adaptive Quantization.
    Tracks usage frequency and quantizes memories based on usage:
    - High usage: 32-bit (full precision)
    - Medium usage: 16-bit
    - Low usage: 8-bit
    """
    
    def __init__(
        self,
        max_memory_size: int = 7,
        usage_threshold: int = 3,  # Number of times a memory must be used to stay in 32-bit
        removal_threshold: int = 1,  # Number of times a memory must be used to not be removed
        quantization_levels: List[int] = [32, 16, 8],  # Bit widths for different usage levels
    ):
        """
        Initialize LRU Quantized Memory.
        
        Args:
            max_memory_size (int): Maximum number of memories to store
            usage_threshold (int): Usage count needed for 32-bit precision
            removal_threshold (int): Usage count below which memories are removed
            quantization_levels (List[int]): Bit widths for different usage levels
        """
        super().__init__(max_memory_size)
        self.usage_threshold = usage_threshold
        self.removal_threshold = removal_threshold
        self.quantization_levels = quantization_levels
        
        # OrderedDict to maintain LRU order
        self._memory_bank = OrderedDict()
        # Track usage frequency for each memory
        self._usage_count: Dict[int, int] = {}
    
    def add_memory(self, frame_idx: int, memory: Dict[str, torch.Tensor]) -> None:
        """
        Add new memory to the bank.
        
        Args:
            frame_idx (int): Index of the frame
            memory (Dict[str, torch.Tensor]): Memory data to store
        """
        # If memory bank is full, remove least recently used
        if self.is_full:
            self._remove_least_used()
        
        # Add new memory with initial usage count
        self._memory_bank[frame_idx] = memory
        self._usage_count[frame_idx] = 0
    
    def _remove_least_used(self) -> None:
        """Remove least recently used memory if usage is below threshold"""
        if not self._memory_bank:
            return
            
        # Find least used memory
        least_used_idx = min(self._usage_count.items(), key=lambda x: x[1])[0]
        
        # Remove if usage is below threshold
        if self._usage_count[least_used_idx] < self.removal_threshold:
            del self._memory_bank[least_used_idx]
            del self._usage_count[least_used_idx]
    
    def get_memory(self, frame_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get memory for a frame, updating usage count and quantization.
        
        Args:
            frame_idx (int): Index of the frame to retrieve
            
        Returns:
            Optional[Dict[str, torch.Tensor]]: Memory data if found, None otherwise
        """
        if frame_idx not in self._memory_bank:
            return None
            
        # Update usage count
        self._usage_count[frame_idx] += 1
        
        # Get current memory
        memory = self._memory_bank[frame_idx]
        
        # On first access (usage_count = 1), return full precision
        if self._usage_count[frame_idx] == 1:
            # Move to end of OrderedDict (most recently used)
            self._memory_bank.move_to_end(frame_idx)
            return memory
        
        # Determine quantization level based on usage
        bits = get_quantization_level(
            self._usage_count[frame_idx],
            [self.usage_threshold, self.removal_threshold]
        )
        
        # Quantize memory tensors
        quantized_memory = {}
        for key, tensor in memory.items():
            if isinstance(tensor, torch.Tensor):
                quantized_memory[key] = quantize_tensor(tensor, bits)
            else:
                quantized_memory[key] = tensor
        
        # Move to end of OrderedDict (most recently used)
        self._memory_bank.move_to_end(frame_idx)
        
        return quantized_memory
    
    def get_all_memories(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Get all memories with their current quantization levels.
        
        Returns:
            Dict[int, Dict[str, torch.Tensor]]: All stored memories
        """
        memories = {}
        for frame_idx in self._memory_bank:
            memories[frame_idx] = self.get_memory(frame_idx)
        return memories
    
    def clear(self) -> None:
        """Clear all memories"""
        self._memory_bank.clear()
        self._usage_count.clear()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory bank.
        
        Returns:
            Dict[str, Any]: Statistics about the memory bank
        """
        return {
            "total_memories": len(self._memory_bank),
            "max_memory_size": self.max_memory_size,
            "usage_counts": self._usage_count.copy(),
            "memory_indices": list(self._memory_bank.keys()),
            "average_usage": sum(self._usage_count.values()) / len(self._usage_count) if self._usage_count else 0
        } 