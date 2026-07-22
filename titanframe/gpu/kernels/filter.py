"""
GPU Predicated Scatter / Filter Kernel
=====================================
Accelerated predicate filtering for GPU arrays.
"""

from typing import Any

try:
    import cupy as cp
except ImportError:
    cp = None

def gpu_filter(array: Any, mask: Any) -> Any:
    """
    Filters a CuPy array or list of arrays using a boolean mask on GPU.
    """
    if cp is None:
        return array[mask]
        
    if isinstance(array, cp.ndarray):
        return array[mask]
    return array[mask]
