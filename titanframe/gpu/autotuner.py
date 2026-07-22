from typing import Dict, Any, Tuple

class KernelAutotuner:
    _cache: Dict[Tuple[str, int], Dict[str, Any]] = {}

    @classmethod
    def get_config(cls, kernel_name: str, num_elements: int) -> Dict[str, int]:
        key = (kernel_name, num_elements)
        if key in cls._cache:
            return cls._cache[key]
        block_size = 256
        if num_elements < 1024:
            block_size = 64
        elif num_elements < 65536:
            block_size = 128
        grid_size = (num_elements + block_size - 1) // block_size
        config = {'block_size': block_size, 'grid_size': grid_size}
        cls._cache[key] = config
        return config
