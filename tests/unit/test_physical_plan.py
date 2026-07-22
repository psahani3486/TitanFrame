"""Unit tests for physical execution plan nodes."""

import pytest
import pyarrow as pa
import titanframe as tf
from titanframe.plan.physical.scan_exec import ScanExec
from titanframe.plan.physical.filter_exec import FilterExec
from titanframe.plan.physical.node import ExecutionContext, Chunk

def test_physical_scan_and_filter():
    table = pa.Table.from_arrays([pa.array([10, 20, 30])], names=["val"])
    ctx = ExecutionContext()
    
    class MockScan(ScanExec):
        def __init__(self, table):
            self.table = table
        def children(self):
            return []
        def execute(self, context):
            for b in self.table.to_batches():
                yield Chunk(b)
                
    scan = MockScan(table)
    filt = FilterExec(scan, tf.col("val") > 15)
    
    results = list(filt.execute(ctx))
    assert len(results) == 1
    assert results[0].data.column("val").to_pylist() == [20, 30]
