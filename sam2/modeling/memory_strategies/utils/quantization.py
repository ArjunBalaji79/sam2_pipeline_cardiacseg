import torch
from typing import Union, List

def quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Quantize tensor to specified bit width.
    
    Args:
        tensor (torch.Tensor): Input tensor to quantize
        bits (int): Target bit width (8, 16, or 32)
        
    Returns:
        torch.Tensor: Quantized tensor
        
    Raises:
        ValueError: If unsupported bit width is provided
    """
    if bits == 32:
        return tensor
    elif bits == 16:
        return tensor.half()
    elif bits == 8:
        # For 8-bit, use linear quantization
        scale = tensor.abs().max() / 127
        return (tensor / scale).round().clamp(-128, 127) * scale
    else:
        raise ValueError(f"Unsupported bit width: {bits}")

def get_quantization_level(usage_count: int, thresholds: List[int]) -> int:
    """
    Determine quantization level based on usage count.
    
    Args:
        usage_count (int): Number of times memory has been accessed
        thresholds (List[int]): List of thresholds for different quantization levels
        
    Returns:
        int: Bit width to use (32, 16, or 8)
    """
    if len(thresholds) != 2:
        raise ValueError("Must provide exactly 2 thresholds")
        
    if usage_count > thresholds[0]:
        return 32
    elif usage_count > thresholds[1]:
        return 16
    else:
        return 8

def dequantize_tensor(tensor: torch.Tensor, original_bits: int) -> torch.Tensor:
    """
    Convert tensor back to original bit width.
    
    Args:
        tensor (torch.Tensor): Quantized tensor
        original_bits (int): Original bit width
        
    Returns:
        torch.Tensor: Dequantized tensor
    """
    if original_bits == 32:
        return tensor.float()
    elif original_bits == 16:
        return tensor.float()
    elif original_bits == 8:
        return tensor.float()
    else:
        raise ValueError(f"Unsupported bit width: {original_bits}") 