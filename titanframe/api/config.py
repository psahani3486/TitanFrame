"""
TitanFrame Global Configuration
=================================

Centralized configuration for execution behavior: GPU device selection,
memory limits, chunk sizes, and spill thresholds.

Example::

    >>> import titanframe as tf
    >>> tf.config.set_gpu_device(0)
    >>> tf.config.set_memory_limit("16GB")
    >>> tf.config.set_chunk_size(131072)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _parse_memory_string(s: str) -> int:
    """Parse human-readable memory strings like '16GB', '512MB', '4TB'."""
    s = s.strip().upper()
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3,
        "TB": 1024 ** 4,
    }
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            return int(float(s[:-len(suffix)].strip()) * mult)
    return int(s)


@dataclass
class TitanFrameConfig:
    """
    Global configuration for TitanFrame.

    Attributes:
        gpu_enabled: Whether to use GPU acceleration. Auto-detected by default.
        gpu_device_id: Default GPU device ID.
        gpu_memory_limit: Maximum GPU memory to use (bytes). None = use all.
        cpu_memory_limit: Maximum host RAM to use (bytes). None = use all.
        nvme_spill_path: Directory for NVMe spill files.
        nvme_spill_limit: Maximum NVMe spill storage (bytes).
        chunk_size: Default number of rows per chunk.
        num_threads: Number of CPU threads for parallel execution.
        spill_threshold: Memory utilization threshold (0.0–1.0) to trigger spilling.
        enable_query_optimizer: Whether to optimize logical plans.
        enable_streaming: Whether to use streaming execution for large datasets.
        verbose: Whether to print execution info.
    """

    gpu_enabled: bool = field(default_factory=lambda: _detect_gpu())
    gpu_device_id: int = 0
    gpu_memory_limit: Optional[int] = None
    gpu_devices: list[int] = field(default_factory=lambda: [0])

    cpu_memory_limit: Optional[int] = None
    num_threads: int = field(default_factory=lambda: os.cpu_count() or 4)

    nvme_spill_path: str = field(default_factory=lambda: os.path.join(
        os.path.expanduser("~"), ".titanframe", "spill"
    ))
    nvme_spill_limit: int = field(default_factory=lambda: 100 * 1024 ** 3)

    chunk_size: int = 65536
    spill_threshold: float = 0.85
    enable_query_optimizer: bool = True
    enable_streaming: bool = True

    verbose: bool = False
    max_display_rows: int = 20
    max_display_columns: int = 50
    max_column_width: int = 40


    def set_gpu_device(self, device_id: int) -> None:
        """Set the default GPU device."""
        self.gpu_device_id = device_id
        self.gpu_enabled = True

    def set_gpu_devices(self, device_ids: list[int]) -> None:
        """Set multiple GPU devices for multi-GPU execution."""
        self.gpu_devices = device_ids
        self.gpu_device_id = device_ids[0]
        self.gpu_enabled = True

    def set_memory_limit(self, limit: str | int) -> None:
        """
        Set CPU memory limit.

        Args:
            limit: Bytes (int) or human-readable string ("16GB", "512MB").
        """
        if isinstance(limit, str):
            self.cpu_memory_limit = _parse_memory_string(limit)
        else:
            self.cpu_memory_limit = limit

    def set_gpu_memory_limit(self, limit: str | int) -> None:
        """Set GPU memory limit."""
        if isinstance(limit, str):
            self.gpu_memory_limit = _parse_memory_string(limit)
        else:
            self.gpu_memory_limit = limit

    def set_chunk_size(self, size: int) -> None:
        """Set the default chunk size (rows per chunk)."""
        if size < 1:
            raise ValueError("Chunk size must be >= 1")
        self.chunk_size = size

    def set_spill_path(self, path: str) -> None:
        """Set the NVMe spill directory."""
        self.nvme_spill_path = path

    def disable_gpu(self) -> None:
        """Force CPU-only execution."""
        self.gpu_enabled = False

    def enable_gpu(self) -> None:
        """Enable GPU acceleration."""
        self.gpu_enabled = True

    def __repr__(self) -> str:
        lines = ["TitanFrameConfig:"]
        lines.append(f"  GPU enabled:      {self.gpu_enabled}")
        lines.append(f"  GPU device(s):    {self.gpu_devices}")
        lines.append(f"  CPU threads:      {self.num_threads}")
        lines.append(f"  Chunk size:       {self.chunk_size:,} rows")
        lines.append(f"  Spill threshold:  {self.spill_threshold:.0%}")
        lines.append(f"  Optimizer:        {'ON' if self.enable_query_optimizer else 'OFF'}")
        if self.cpu_memory_limit:
            lines.append(f"  CPU memory limit: {self.cpu_memory_limit / 1024**3:.1f} GB")
        if self.gpu_memory_limit:
            lines.append(f"  GPU memory limit: {self.gpu_memory_limit / 1024**3:.1f} GB")
        return "\n".join(lines)


def _detect_gpu() -> bool:
    """Detect if a CUDA-capable GPU is available."""
    try:
        import cupy
        return True
    except (ImportError, Exception):
        return False



config = TitanFrameConfig()
