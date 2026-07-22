"""
Database I/O
============
Read and write SQL databases using SQLAlchemy.
"""
from titanframe.plan.logical.scan import Scan, ScanFormat

def read_sql(query: str, uri: str, **kwargs) -> Scan:
    """Read from a SQL database into a logical plan."""
    import sqlalchemy as sa
    import pandas as pd
    from titanframe.core.schema import Schema
    import pyarrow as pa
    
    engine = sa.create_engine(uri)
    if "SELECT" in query.upper():
        q = f"SELECT * FROM ({query}) AS sub LIMIT 0"
    else:
        q = f"SELECT * FROM {query} LIMIT 0"
        
    with engine.connect() as conn:
        df = pd.read_sql_query(sa.text(q), conn)
        schema = Schema.from_arrow(pa.Schema.from_pandas(df))
        
    source = f"{uri}::{query}"
    return Scan(source=source, format=ScanFormat.SQL, schema=schema, **kwargs)

def write_sql(df, table: str, uri: str, **kwargs):
    """Write a DataFrame to a SQL database."""
    import sqlalchemy as sa
    import pyarrow as pa
    
    engine = sa.create_engine(uri)
    
    from titanframe.plan.physical.node import ExecutionContext
    
    context = ExecutionContext()
    phys_plan = df._get_physical_plan()
    
    with engine.connect() as conn:
        for chunk in phys_plan.execute(context):
            chunk_df = chunk.data.to_pandas()
            chunk_df.to_sql(table, conn, if_exists='append', index=False, **kwargs)
