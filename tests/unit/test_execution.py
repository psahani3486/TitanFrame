import pytest
import pyarrow as pa
import pathlib
import os

from titanframe.core.dtypes import Int64, Utf8, Bool
from titanframe.core.schema import Schema
from titanframe.expr.column_expr import col
from titanframe.expr.literal_expr import lit
from titanframe.plan.logical.scan import Scan, ScanFormat
from titanframe.plan.logical.projection import Projection
from titanframe.plan.logical.filter import Filter
from titanframe.plan.logical.limit import Limit
from titanframe.plan.physical.planner import PhysicalPlanner
from titanframe.engine.scheduler import DAGScheduler
from titanframe.plan.physical.node import ExecutionContext

@pytest.fixture
def mock_parquet(tmp_path):
    import pyarrow.parquet as pq
    path = tmp_path / "mock.parquet"
    table = pa.table({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 40, 45]
    })
    pq.write_table(table, str(path))
    return str(path)

def test_scan_and_project(mock_parquet):
    schema = Schema({"id": Int64, "name": Utf8, "age": Int64})
    scan = Scan(mock_parquet, ScanFormat.PARQUET, schema)
    
    # Select id, age + 5
    proj = Projection(scan, [col("id"), col("age") + lit(5)])
    
    planner = PhysicalPlanner()
    phys_plan = planner.plan(proj)
    
    ctx = ExecutionContext()
    scheduler = DAGScheduler()
    table = scheduler.execute(phys_plan, ctx)
    
    assert table.num_rows == 5
    assert table.num_columns == 2
    assert table.column(1).to_pylist() == [30, 35, 40, 45, 50]

def test_filter(mock_parquet):
    schema = Schema({"id": Int64, "name": Utf8, "age": Int64})
    scan = Scan(mock_parquet, ScanFormat.PARQUET, schema)
    
    # Filter age > 30
    filt = Filter(scan, col("age") > lit(30))
    
    planner = PhysicalPlanner()
    phys_plan = planner.plan(filt)
    
    ctx = ExecutionContext()
    scheduler = DAGScheduler()
    table = scheduler.execute(phys_plan, ctx)
    
    assert table.num_rows == 3
    assert table.column("age").to_pylist() == [35, 40, 45]

def test_limit(mock_parquet):
    schema = Schema({"id": Int64, "name": Utf8, "age": Int64})
    scan = Scan(mock_parquet, ScanFormat.PARQUET, schema)
    
    # Limit 2 offset 1 -> should return [2, 3]
    limit = Limit(scan, limit=2, offset=1)
    
    planner = PhysicalPlanner()
    phys_plan = planner.plan(limit)
    
    ctx = ExecutionContext()
    scheduler = DAGScheduler()
    table = scheduler.execute(phys_plan, ctx)
    
    assert table.num_rows == 2
    assert table.column("id").to_pylist() == [2, 3]

def test_memory_limited_aggregation(mock_parquet):
    from titanframe.plan.logical.aggregation import Aggregation
    from titanframe.memory.manager import MemoryManager
    from titanframe.expr.agg_expr import sum_
    
    schema = Schema({"id": Int64, "name": Utf8, "age": Int64})
    # Scan with small batch size to ensure multiple chunks
    scan = Scan(mock_parquet, ScanFormat.PARQUET, schema)
    
    # Simple agg: sum(age) group by id
    agg = Aggregation(scan, [col("id")], [sum_(col("age"))])
    
    planner = PhysicalPlanner()
    phys_plan = planner.plan(agg)
    
    # Extremely small budget to force spills during chunk collection
    # A chunk of 5 rows is ~200 bytes, we set budget to 150 bytes
    manager = MemoryManager(budget_bytes=150)
    # Set small batch_size so it actually breaks into multiple chunks
    ctx = ExecutionContext(batch_size=2, memory_manager=manager)
    
    scheduler = DAGScheduler()
    table = scheduler.execute(phys_plan, ctx)
    
    assert table is not None
    # Just asserting it didn't crash and returned results
    assert table.num_rows == 5
    
    manager.cleanup()
