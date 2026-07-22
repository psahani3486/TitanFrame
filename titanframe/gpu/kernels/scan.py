"""
GPU Parallel Prefix Scan Kernel
===============================
Prefix sum (cumsum) kernel implementation for GPU buffers.
"""

from typing import Any

try:
    import cupy as cp
except ImportError:
    cp = None

def gpu_prefix_scan(arr: Any) -> Any:
    """Computes inclusive prefix sum (cumsum) on GPU."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.cumsum(arr)
    import numpy as np
    return np.cumsum(arr)
