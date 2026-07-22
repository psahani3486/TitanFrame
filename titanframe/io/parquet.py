"""
Parquet I/O
===========
Read and write Parquet files with predicate/projection pushdown.
"""
from titanframe.plan.logical.scan import Scan, ScanFormat
import pyarrow.parquet as pq

def read_parquet(source: str, **kwargs) -> Scan:
    """Read a Parquet file into a logical plan."""
    meta = pq.read_metadata(source)
    from titanframe.core.schema import Schema
    schema = Schema.from_arrow(meta.schema.to_arrow_schema())
    
    return Scan(source=source, format=ScanFormat.PARQUET, schema=schema, **kwargs)

def write_parquet(df, target: str, **kwargs):
    """Write a DataFrame to a Parquet file."""
    import pyarrow.parquet as pq
    
    from titanframe.plan.physical.node import ExecutionContext
    from titanframe.engine.scheduler import DAGScheduler
    
    context = ExecutionContext()
    scheduler = DAGScheduler()
    
    writer = None
    
    phys_plan = df._get_physical_plan()
    
    for chunk in phys_plan.execute(context):
        if writer is None:
            writer = pq.ParquetWriter(target, chunk.data.schema, **kwargs)
        writer.write_batch(chunk.data)
        
    if writer:
        writer.close()
