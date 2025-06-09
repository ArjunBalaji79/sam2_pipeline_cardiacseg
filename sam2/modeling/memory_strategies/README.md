# Memory Strategies Pipeline

This directory contains the implementation of memory management strategies for the SAM2 (Segment Anything Model 2) video prediction system. The pipeline provides a flexible framework for managing memory banks with different strategies.

## Directory Structure

```
memory_strategies/
├── base.py              # Base class for memory strategies
├── strategies/          # Implemented memory strategies
│   ├── lru_quantized.py # LRU with adaptive quantization
│   └── __init__.py
├── utils/              # Utility functions
│   ├── quantization.py # Quantization utilities
│   └── __init__.py
├── tests/              # Unit tests
├── experiments/        # Example usage and comparisons
└── README.md          # This file
```

## Components

### Base Memory Strategy
- `BaseMemoryStrategy`: Abstract base class defining the interface for all memory strategies
- Key methods:
  - `add_memory()`: Add new memory to the bank
  - `get_memory()`: Retrieve memory for a specific frame
  - `get_all_memories()`: Get all stored memories
  - `clear()`: Clear the memory bank
  - `get_memory_stats()`: Get memory bank statistics

### Implemented Strategies

#### LRU Quantized Memory
- Implements Least Recently Used (LRU) with adaptive quantization
- Features:
  - Tracks memory usage frequency
  - Adaptive quantization based on usage:
    - High usage: 32-bit (full precision)
    - Medium usage: 16-bit
    - Low usage: 8-bit
  - Configurable thresholds for quantization and removal

### Utility Functions
- Quantization utilities for memory compression
- Memory statistics and monitoring tools

## Usage

### Basic Usage

```python
from sam2.modeling.memory_strategies.strategies.lru_quantized import LRUQuantizedMemory

# Initialize memory strategy
memory = LRUQuantizedMemory(
    max_memory_size=7,
    usage_threshold=3,
    removal_threshold=1
)

# Add memory
memory.add_memory(frame_idx=0, memory={
    "features": tensor,
    "mask": mask_tensor,
    "metadata": metadata
})

# Retrieve memory
retrieved = memory.get_memory(frame_idx=0)

# Get statistics
stats = memory.get_memory_stats()
```

### Integration with SAM2

```python
from sam2.modeling.sam2_memory_predictor import SAM2MemoryPredictor

# Initialize predictor with memory strategy
predictor = SAM2MemoryPredictor(
    memory_strategy="lru_quantized",
    memory_strategy_kwargs={
        "max_memory_size": 7,
        "usage_threshold": 3,
        "removal_threshold": 1
    }
)
```

## Memory Bank Size

By default, the memory bank stores 7 frames:
- 1 input frame
- 6 previous frames

This can be configured through the `max_memory_size` parameter.

## Memory Management

The pipeline manages memory through:
1. **Addition**: New memories are added with initial usage count of 0
2. **Retrieval**: 
   - Updates usage count
   - Applies adaptive quantization
   - Moves memory to most recently used position
3. **Removal**: 
   - Removes least used memories when bank is full
   - Uses configurable thresholds for removal decisions

## Experiments

The `experiments/` directory contains example scripts demonstrating:
- Different memory strategy configurations
- Performance comparisons
- Memory usage patterns
- Quantization effects

See individual experiment files for specific examples and comparisons.

## Best Practices

1. **Memory Size**: Choose appropriate `max_memory_size` based on:
   - Available memory
   - Temporal context needed
   - Performance requirements

2. **Quantization Thresholds**: Configure based on:
   - Memory constraints
   - Precision requirements
   - Usage patterns

3. **Monitoring**: Use `get_memory_stats()` to:
   - Track memory usage
   - Monitor performance
   - Debug issues

## Contributing

To add a new memory strategy:
1. Create a new class inheriting from `BaseMemoryStrategy`
2. Implement all required methods
3. Add tests in the `tests/` directory
4. Add example usage in `experiments/` 