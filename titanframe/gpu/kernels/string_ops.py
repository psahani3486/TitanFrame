"""
GPU String Processing Kernels
=============================
Custom GPU string operations for elementwise string filtering, length, and substring matching.
"""

from typing import Any, List

try:
    import cupy as cp
except ImportError:
    cp = None

def gpu_string_length(strings: List[str]) -> Any:
    """Compute string lengths on GPU / CPU fallback."""
    lengths = [len(s) for s in strings]
    if cp is not None:
        return cp.array(lengths, dtype=cp.int32)
    return lengths

def gpu_string_contains(strings: List[str], pattern: str) -> Any:
    """Compute boolean substring containment vector."""
    matches = [pattern in s for s in strings]
    if cp is not None:
        return cp.array(matches, dtype=cp.bool_)
    return matches
