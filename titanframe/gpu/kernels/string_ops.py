from typing import Any, List
try:
    import cupy as cp
except ImportError:
    cp = None

def gpu_string_length(strings: List[str]) -> Any:
    lengths = [len(s) for s in strings]
    if cp is not None:
        return cp.array(lengths, dtype=cp.int32)
    return lengths

def gpu_string_contains(strings: List[str], pattern: str) -> Any:
    matches = [pattern in s for s in strings]
    if cp is not None:
        return cp.array(matches, dtype=cp.bool_)
    return matches
