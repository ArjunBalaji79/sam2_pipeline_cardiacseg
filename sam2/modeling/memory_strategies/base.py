from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
import torch

class BaseMemoryStrategy(ABC):
    """
    Base class for all memory bank strategies.
    All memory strategies must implement these methods.
    """
    
    def __init__(self, max_memory_size: int = 7):
        """
        Initialize the memory strategy.
        
        Args:
            max_memory_size (int): Maximum number of memories to store
        """
        self.max_memory_size = max_memory_size
        self._memory_bank: Dict[int, Dict[str, torch.Tensor]] = {}
    
    @abstractmethod
    def add_memory(self, frame_idx: int, memory: Dict[str, torch.Tensor]) -> None:
        """
        Add new memory to the bank.
        
        Args:
            frame_idx (int): Index of the frame
            memory (Dict[str, torch.Tensor]): Memory data to store
        """
        pass
    
    @abstractmethod
    def get_memory(self, frame_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get memory for a specific frame.
        
        Args:
            frame_idx (int): Index of the frame to retrieve
            
        Returns:
            Optional[Dict[str, torch.Tensor]]: Memory data if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_all_memories(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Get all memories in the bank.
        
        Returns:
            Dict[int, Dict[str, torch.Tensor]]: All stored memories
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all memories from the bank"""
        pass
    
    @abstractmethod
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory bank.
        
        Returns:
            Dict[str, Any]: Statistics about the memory bank
        """
        pass
    
    @property
    def memory_size(self) -> int:
        """Get current number of memories in the bank"""
        return len(self._memory_bank)
    
    @property
    def is_full(self) -> bool:
        """Check if memory bank is full"""
        return self.memory_size >= self.max_memory_size 