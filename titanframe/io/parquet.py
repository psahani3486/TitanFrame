from titanframe.plan.logical.scan import Scan, ScanFormat
import pyarrow.parquet as pq

def read_parquet(source: str, **kwargs) -> Scan:
    meta = pq.read_metadata(source)
    from titanframe.core.schema import Schema
    schema = Schema.from_arrow(meta.schema.to_arrow_schema())
    return Scan(source=source, format=ScanFormat.PARQUET, schema=schema, **kwargs)

def write_parquet(df, target: str, **kwargs):
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
