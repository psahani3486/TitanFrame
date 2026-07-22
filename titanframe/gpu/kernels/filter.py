from typing import Any
try:
    import cupy as cp
except ImportError:
    cp = None

def gpu_filter(array: Any, mask: Any) -> Any:
    if cp is None:
        return array[mask]
    if isinstance(array, cp.ndarray):
        return array[mask]
    return array[mask]
