"""
Parquet I/O
===========
Read and write Parquet files with predicate/projection pushdown.
"""
from titanframe.plan.logical.scan import Scan, ScanFormat
import pyarrow.parquet as pq

def read_parquet(source: str, **kwargs) -> Scan:
    """Read a Parquet file into a logical plan."""
    # We inspect the schema directly from the Parquet file to inform the logical plan.
    meta = pq.read_metadata(source)
    # PyArrow Schema to TitanFrame schema could be handled in a wrapper, 
    # but for simplicity, we pass None and let execution handle it, or we can use from_arrow
    from titanframe.core.schema import Schema
    schema = Schema.from_arrow(meta.schema.to_arrow_schema())
    
    return Scan(source=source, format=ScanFormat.PARQUET, schema=schema, **kwargs)

def write_parquet(df, target: str, **kwargs):
    """Write a DataFrame to a Parquet file."""
    # Execute the dataframe and stream chunks to disk
    import pyarrow.parquet as pq
    
    # We assume df is a LazyFrame or DataFrame with an .execute() or .collect()
    # Actually, df can yield chunks via scheduler.
    from titanframe.plan.physical.node import ExecutionContext
    from titanframe.engine.scheduler import DAGScheduler
    
    context = ExecutionContext()
    scheduler = DAGScheduler()
    
    writer = None
    
    # df.logical_plan gives the plan. df._get_physical_plan() gives the physical plan.
    # To stream, we can use plan.execute()
    phys_plan = df._get_physical_plan()
    
    for chunk in phys_plan.execute(context):
        if writer is None:
            writer = pq.ParquetWriter(target, chunk.data.schema, **kwargs)
        writer.write_batch(chunk.data)
        
    if writer:
        writer.close()
