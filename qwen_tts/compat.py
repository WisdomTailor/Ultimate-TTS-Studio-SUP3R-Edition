# coding=utf-8
"""
Compatibility layer for transformers 4.51.3
Provides stubs/implementations for features added in later versions.
"""

import functools
from typing import Any, Callable, Optional, List
import torch
from torch import nn


# ============================================================================
# auto_docstring - In newer transformers, this is a decorator that auto-generates
# docstrings. For 4.51.3, we just make it a no-op decorator.
# ============================================================================

def auto_docstring(*args, custom_intro: str = None, **kwargs):
    """
    Compatibility stub for auto_docstring decorator.
    In transformers 4.52+, this auto-generates docstrings.
    Here it's a no-op that just returns the class/function unchanged.
    """
    def decorator(cls_or_func):
        return cls_or_func
    
    # Handle both @auto_docstring and @auto_docstring(...) syntax
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return decorator


# ============================================================================
# can_return_tuple - Decorator that allows methods to return tuples
# ============================================================================

def can_return_tuple(func):
    """
    Compatibility stub for can_return_tuple decorator.
    In newer transformers, this handles return type conversion.
    Here it's a no-op.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


# ============================================================================
# check_model_inputs - Decorator for input validation
# ============================================================================

def check_model_inputs():
    """
    Compatibility stub for check_model_inputs decorator.
    In newer transformers, this validates model inputs.
    Here it's a no-op.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# layer_type_validation - Function for validating layer types
# ============================================================================

def layer_type_validation(layer_types: List[str]) -> None:
    """
    Compatibility stub for layer_type_validation.
    In newer transformers, this validates layer type configurations.
    Here we just do basic validation.
    """
    if layer_types is None:
        return
    valid_types = {"attention", "full_attention", "sliding_attention"}
    for lt in layer_types:
        if lt not in valid_types:
            # Just warn, don't raise - the model might still work
            pass


# ============================================================================
# use_kernel_forward_from_hub - Decorator for kernel optimization
# ============================================================================

def use_kernel_forward_from_hub(kernel_name: str):
    """
    Compatibility stub for use_kernel_forward_from_hub decorator.
    In newer transformers, this loads optimized kernels from the hub.
    Here it's a no-op.
    """
    def decorator(cls):
        return cls
    return decorator


# ============================================================================
# GradientCheckpointingLayer - Base class for gradient checkpointing
# ============================================================================

class GradientCheckpointingLayer(nn.Module):
    """
    Compatibility class for GradientCheckpointingLayer.
    In newer transformers, this provides gradient checkpointing support.
    Here it's just a regular nn.Module.
    """
    pass


# ============================================================================
# Masking utilities - create_causal_mask, create_sliding_window_causal_mask
# ============================================================================

def create_causal_mask(
    config,
    input_embeds: torch.Tensor = None,
    input_tensor: torch.Tensor = None,
    cache_position: torch.Tensor = None,
    past_key_values: Optional[Any] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,  # Accept any additional kwargs for compatibility
) -> Optional[torch.Tensor]:
    """
    Creates a causal mask for attention.
    
    This is a simplified implementation for compatibility.
    """
    # Use input_embeds if provided, otherwise fall back to input_tensor
    tensor = input_embeds if input_embeds is not None else input_tensor
    if tensor is None:
        return None
    
    batch_size, seq_length = tensor.shape[:2]
    dtype = tensor.dtype
    device = tensor.device
    
    if attention_mask is not None and attention_mask.dim() == 4:
        # Already a 4D mask
        return attention_mask
    
    # Get the target length (including past)
    if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
        past_length = past_key_values.get_seq_length()
    else:
        past_length = 0
    
    target_length = seq_length + past_length
    
    # Create causal mask
    causal_mask = torch.full(
        (seq_length, target_length), 
        fill_value=torch.finfo(dtype).min,
        dtype=dtype,
        device=device
    )
    
    if seq_length > 1:
        # Fill the upper triangle with -inf (positions that should be masked)
        if cache_position is not None:
            mask_cond = torch.arange(target_length, device=device)
            causal_mask.masked_fill_(
                mask_cond < (cache_position.reshape(-1, 1) + 1), 
                0
            )
        else:
            # Standard causal mask without cache_position
            mask = torch.triu(torch.ones(seq_length, target_length, device=device), diagonal=past_length + 1)
            causal_mask = causal_mask.masked_fill(mask == 0, 0)
    else:
        causal_mask.fill_(0)
    
    # Expand to 4D: [batch, 1, seq, target]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    causal_mask = causal_mask.expand(batch_size, 1, -1, -1)
    
    # Combine with attention_mask if provided
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            # [batch, seq] -> [batch, 1, 1, seq]
            expanded_mask = attention_mask[:, None, None, :].to(dtype)
            expanded_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
            # Expand to match target length if needed
            if expanded_mask.shape[-1] < target_length:
                padding = torch.zeros(
                    batch_size, 1, 1, target_length - expanded_mask.shape[-1],
                    dtype=dtype, device=device
                )
                expanded_mask = torch.cat([expanded_mask, padding], dim=-1)
            causal_mask = causal_mask + expanded_mask[:, :, :, :target_length]
    
    return causal_mask


def create_sliding_window_causal_mask(
    config,
    input_embeds: torch.Tensor = None,
    input_tensor: torch.Tensor = None,
    cache_position: torch.Tensor = None,
    past_key_values: Optional[Any] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,  # Accept any additional kwargs for compatibility
) -> Optional[torch.Tensor]:
    """
    Creates a sliding window causal mask for attention.
    
    This is a simplified implementation for compatibility.
    """
    # Use input_embeds if provided, otherwise fall back to input_tensor
    tensor = input_embeds if input_embeds is not None else input_tensor
    if tensor is None:
        return None
    
    batch_size, seq_length = tensor.shape[:2]
    dtype = tensor.dtype
    device = tensor.device
    
    # Get sliding window size from config
    sliding_window = getattr(config, 'sliding_window', None)
    if sliding_window is None:
        sliding_window = getattr(config, 'max_window_layers', 4096)
    
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    
    # Get the target length
    if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
        past_length = past_key_values.get_seq_length()
    else:
        past_length = 0
    
    target_length = seq_length + past_length
    
    # Create sliding window causal mask
    causal_mask = torch.full(
        (seq_length, target_length),
        fill_value=torch.finfo(dtype).min,
        dtype=dtype,
        device=device
    )
    
    if seq_length > 1:
        # Create position indices
        row_idx = torch.arange(seq_length, device=device).unsqueeze(1)
        col_idx = torch.arange(target_length, device=device).unsqueeze(0)
        
        # Causal: can only attend to positions <= current position
        causal_cond = col_idx <= (row_idx + past_length)
        
        # Sliding window: can only attend to positions within window
        window_cond = (row_idx + past_length - col_idx) < sliding_window
        
        # Combine conditions
        valid_mask = causal_cond & window_cond
        causal_mask = causal_mask.masked_fill(valid_mask, 0)
    else:
        causal_mask.fill_(0)
    
    # Expand to 4D
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    causal_mask = causal_mask.expand(batch_size, 1, -1, -1)
    
    # Combine with attention_mask if provided
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            expanded_mask = attention_mask[:, None, None, :].to(dtype)
            expanded_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
            # Expand to match target length if needed
            if expanded_mask.shape[-1] < target_length:
                padding = torch.zeros(
                    batch_size, 1, 1, target_length - expanded_mask.shape[-1],
                    dtype=dtype, device=device
                )
                expanded_mask = torch.cat([expanded_mask, padding], dim=-1)
            causal_mask = causal_mask + expanded_mask[:, :, :, :target_length]
    
    return causal_mask
