"""
JSON I/O
========
Read and write NDJSON files.
"""
from titanframe.plan.logical.scan import Scan, ScanFormat

def read_json(source: str, **kwargs) -> Scan:
    """Read a JSON file into a logical plan."""
    # We inspect the schema directly from the JSON file by reading the first few lines
    import pyarrow.json as pj
    from titanframe.core.schema import Schema
    
    # Simple schema inference by parsing a small block
    table = pj.read_json(source, read_options=pj.ReadOptions(block_size=1024*1024))
    schema = Schema.from_arrow(table.schema)
    
    return Scan(source=source, format=ScanFormat.JSON, schema=schema, **kwargs)

def write_json(df, target: str, **kwargs):
    """Write a DataFrame to an NDJSON file."""
    import pyarrow.json as pj
    
    from titanframe.plan.physical.node import ExecutionContext
    from titanframe.engine.scheduler import DAGScheduler
    
    context = ExecutionContext()
    phys_plan = df._get_physical_plan()
    
    # PyArrow doesn't have a streaming JSON writer yet, we must collect or stream to strings
    # For now, collect and write (or manually encode JSON strings)
    table = df.collect()
    
    # Workaround since PyArrow lacks native write_json
    import json
    with open(target, 'w') as f:
        for batch in table.to_batches():
            d = batch.to_pydict()
            keys = list(d.keys())
            for i in range(batch.num_rows):
                row = {k: d[k][i] for k in keys}
                f.write(json.dumps(row) + "\n")
