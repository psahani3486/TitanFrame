"""
ScanExec Node
=============

Reads data from source in chunks.
"""
from typing import Iterator, Optional
import pyarrow.parquet as pq
import pyarrow.csv as csv
import pyarrow as pa

from titanframe.plan.physical.node import PhysicalPlan, ExecutionContext, Chunk
import pyarrow.dataset as ds
from titanframe.expr.base import Expr
from titanframe.plan.logical.scan import ScanFormat

class ScanExec(PhysicalPlan):
    def __init__(
        self,
        source: str,
        format: ScanFormat,
        columns: Optional[list[str]] = None,
        limit: Optional[int] = None,
        filters: Optional[Expr] = None,
        table = None,
    ):
        self.source = source
        self.format = format
        self.columns = columns
        self.limit = limit
        self.filters = filters
        self.table = table
        
    def execute(self, context: ExecutionContext) -> Iterator[Chunk]:
        rows_yielded = 0
        
        if self.format == ScanFormat.IN_MEMORY:
            if self.table is None:
                raise ValueError("IN_MEMORY scan requires a valid table.")
            for chunk in self.table._chunks:
                # Basic limit support
                if self.limit is not None:
                    remaining = self.limit - rows_yielded
                    if remaining <= 0: break
                    if chunk.num_rows > remaining:
                        chunk = chunk.slice(0, remaining)
                
                rows_yielded += chunk.num_rows
                yield chunk
                
                if self.limit is not None and rows_yielded >= self.limit:
                    break
        elif self.format == ScanFormat.PARQUET:
            # For predicate pushdown, convert filters to pyarrow expressions if possible
            ds_filter = None
            if self.filters is not None:
                from titanframe.plan.physical.evaluator import ExprEvaluator
                evaluator = ExprEvaluator()
                try:
                    ds_filter = evaluator._compile(self.filters)
                except NotImplementedError:
                    pass
                    
            dataset = ds.dataset(self.source, format="parquet")
            for batch in dataset.to_batches(batch_size=context.batch_size, columns=self.columns, filter=ds_filter):
                if self.limit is not None:
                    remaining = self.limit - rows_yielded
                    if remaining <= 0:
                        break
                    if batch.num_rows > remaining:
                        batch = batch.slice(0, remaining)
                
                rows_yielded += batch.num_rows
                yield Chunk(batch)
                
                if self.limit is not None and rows_yielded >= self.limit:
                    break
                    
        elif self.format == ScanFormat.JSON or self.format == ScanFormat.NDJSON:
            import pyarrow.json as pj
            
            table = pj.read_json(self.source)
            if self.columns:
                table = table.select(self.columns)
            batches = table.to_batches(context.batch_size)
            for batch in batches:
                if self.limit is not None:
                    remaining = self.limit - rows_yielded
                    if remaining <= 0: break
                    if batch.num_rows > remaining:
                        batch = batch.slice(0, remaining)
                rows_yielded += batch.num_rows
                yield Chunk(batch)
                if self.limit is not None and rows_yielded >= self.limit:
                    break
                        
        elif self.format == ScanFormat.DATABASE or self.format == ScanFormat.SQL:
            import sqlalchemy as sa
            import pandas as pd
            
            uri, _, query_or_table = self.source.partition("::")
            if not query_or_table:
                query_or_table = "default_table"
                
            engine = sa.create_engine(uri)
            
            if "SELECT" in query_or_table.upper():
                q = query_or_table
            else:
                cols = "*" if not self.columns else ", ".join(self.columns)
                q = f"SELECT {cols} FROM {query_or_table}"
                
            try:
                with engine.connect() as conn:
                    for chunk_df in pd.read_sql_query(sa.text(q), conn, chunksize=context.batch_size):
                        batch = pa.RecordBatch.from_pandas(chunk_df)
                        if self.limit is not None:
                            remaining = self.limit - rows_yielded
                            if remaining <= 0: break
                            if batch.num_rows > remaining:
                                batch = batch.slice(0, remaining)
                        rows_yielded += batch.num_rows
                        yield Chunk(batch)
                        if self.limit is not None and rows_yielded >= self.limit:
                            break
            finally:
                engine.dispose()
                    
        elif self.format == ScanFormat.CSV:
            # Note: pyarrow.csv doesn't allow chunk size natively without read_options tweaking
            # but we can use open_csv and iterate.
            with csv.open_csv(self.source) as reader:
                for batch in reader:
                    if self.columns:
                        arrays = [batch.column(c) for c in self.columns]
                        batch = pa.RecordBatch.from_arrays(arrays, names=self.columns)
                        
                    if self.limit is not None:
                        remaining = self.limit - rows_yielded
                        if remaining <= 0:
                            break
                        if batch.num_rows > remaining:
                            batch = batch.slice(0, remaining)
                            
                    rows_yielded += batch.num_rows
                    yield Chunk(batch)
                    
                    if self.limit is not None and rows_yielded >= self.limit:
                        break
        else:
            raise NotImplementedError(f"Format {self.format} not supported for scan")
