import pytest
import os
import tempfile
import pyarrow as pa

from titanframe.io.parquet import read_parquet, write_parquet
from titanframe.io.json import read_json, write_json
from titanframe.io.database import read_sql, write_sql
from titanframe.api.lazyframe import LazyFrame
from titanframe.plan.logical.scan import Scan, ScanFormat
from titanframe.engine.scheduler import DAGScheduler
from titanframe.plan.physical.planner import PhysicalPlanner
from titanframe.plan.physical.node import ExecutionContext

@pytest.fixture
def sample_table():
    return pa.table({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 40, 45]
    })

def test_parquet_roundtrip(sample_table):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.parquet")
        
        # Write
        # Using pyarrow directly for write to mock the 'df' since our write_parquet expects a df with _get_physical_plan
        # Actually, let's just create a mock dataframe wrapper or use pyarrow to write and then test read_parquet
        import pyarrow.parquet as pq
        pq.write_table(sample_table, path)
        
        # Read
        scan_plan = read_parquet(path)
        assert isinstance(scan_plan, Scan)
        assert scan_plan.format == ScanFormat.PARQUET
        
        planner = PhysicalPlanner()
        phys_plan = planner.plan(scan_plan)
        
        ctx = ExecutionContext()
        scheduler = DAGScheduler()
        table = scheduler.execute(phys_plan, ctx)
        
        assert table.num_rows == 5
        assert table.column("name").to_pylist() == ["Alice", "Bob", "Charlie", "David", "Eve"]

def test_json_roundtrip(sample_table):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.json")
        
        # Write using standard pyarrow json if possible, or standard json
        import json
        with open(path, 'w') as f:
            for row in sample_table.to_pylist():
                f.write(json.dumps(row) + "\n")
                
        # Read
        scan_plan = read_json(path)
        assert isinstance(scan_plan, Scan)
        assert scan_plan.format == ScanFormat.JSON
        
        planner = PhysicalPlanner()
        phys_plan = planner.plan(scan_plan)
        
        ctx = ExecutionContext()
        scheduler = DAGScheduler()
        table = scheduler.execute(phys_plan, ctx)
        
        assert table.num_rows == 5
        assert table.column("name").to_pylist() == ["Alice", "Bob", "Charlie", "David", "Eve"]

def test_sql_roundtrip(sample_table):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        uri = f"sqlite:///{db_path}"
        
        import pandas as pd
        import sqlalchemy as sa
        engine = sa.create_engine(uri)
        df = sample_table.to_pandas()
        with engine.connect() as conn:
            df.to_sql("users", conn, index=False)
        engine.dispose()
            
        # Read
        scan_plan = read_sql("users", uri)
        assert isinstance(scan_plan, Scan)
        assert scan_plan.format == ScanFormat.SQL
        
        planner = PhysicalPlanner()
        phys_plan = planner.plan(scan_plan)
        
        ctx = ExecutionContext()
        scheduler = DAGScheduler()
        table = scheduler.execute(phys_plan, ctx)
        
        assert table.num_rows == 5
        assert table.column("name").to_pylist() == ["Alice", "Bob", "Charlie", "David", "Eve"]
