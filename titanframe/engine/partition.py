from abc import ABC, abstractmethod
from typing import List
import pyarrow as pa
import pyarrow.compute as pc

class PartitionStrategy(ABC):

    @abstractmethod
    def partition(self, batch: pa.RecordBatch, num_partitions: int) -> List[pa.RecordBatch]:
        pass

class HashPartition(PartitionStrategy):

    def __init__(self, key_columns: List[str]):
        self.key_columns = key_columns

    def partition(self, batch: pa.RecordBatch, num_partitions: int) -> List[pa.RecordBatch]:
        if num_partitions <= 1 or batch.num_rows == 0:
            return [batch]
        hashes = None
        for col_name in self.key_columns:
            arr = batch.column(col_name)
            h = pc.hash_expression(arr) if hasattr(pc, 'hash_expression') else pc.utf8_length(arr) if pa.types.is_string(arr.type) else arr
            hashes = h if hashes is None else pc.add(hashes, h)
        partition_indices = pc.mod(pc.abs(hashes), pa.scalar(num_partitions))
        results = []
        for p in range(num_partitions):
            mask = pc.equal(partition_indices, p)
            sliced = batch.filter(mask)
            results.append(sliced)
        return results

class RoundRobinPartition(PartitionStrategy):

    def partition(self, batch: pa.RecordBatch, num_partitions: int) -> List[pa.RecordBatch]:
        if num_partitions <= 1 or batch.num_rows == 0:
            return [batch]
        results = []
        indices = pa.array(list(range(batch.num_rows)))
        for p in range(num_partitions):
            mask = pc.equal(pc.mod(indices, pa.scalar(num_partitions)), p)
            results.append(batch.filter(mask))
        return results

class RangePartition(PartitionStrategy):

    def __init__(self, column: str, boundaries: List[float]):
        self.column = column
        self.boundaries = boundaries

    def partition(self, batch: pa.RecordBatch, num_partitions: int) -> List[pa.RecordBatch]:
        if num_partitions <= 1 or batch.num_rows == 0:
            return [batch]
        arr = batch.column(self.column)
        results = []
        lower = None
        for i in range(num_partitions):
            upper = self.boundaries[i] if i < len(self.boundaries) else None
            if lower is None and upper is not None:
                mask = pc.less(arr, upper)
            elif lower is not None and upper is not None:
                mask = pc.and_(pc.greater_equal(arr, lower), pc.less(arr, upper))
            elif lower is not None and upper is None:
                mask = pc.greater_equal(arr, lower)
            else:
                mask = pa.array([True] * batch.num_rows)
            results.append(batch.filter(mask))
            lower = upper
        return results
