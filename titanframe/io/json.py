from titanframe.plan.logical.scan import Scan, ScanFormat

def read_json(source: str, **kwargs) -> Scan:
    import pyarrow.json as pj
    from titanframe.core.schema import Schema
    table = pj.read_json(source, read_options=pj.ReadOptions(block_size=1024 * 1024))
    schema = Schema.from_arrow(table.schema)
    return Scan(source=source, format=ScanFormat.JSON, schema=schema, **kwargs)

def write_json(df, target: str, **kwargs):
    import pyarrow.json as pj
    from titanframe.plan.physical.node import ExecutionContext
    from titanframe.engine.scheduler import DAGScheduler
    context = ExecutionContext()
    phys_plan = df._get_physical_plan()
    table = df.collect()
    import json
    with open(target, 'w') as f:
        for batch in table.to_batches():
            d = batch.to_pydict()
            keys = list(d.keys())
            for i in range(batch.num_rows):
                row = {k: d[k][i] for k in keys}
                f.write(json.dumps(row) + '\n')
