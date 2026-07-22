"""
Scan Node — Data Source
========================

The Scan node is the leaf of every logical plan. It represents reading
data from an external source (CSV, Parquet, Arrow IPC, database).

Scan supports **pushed-down** predicates and projections: the optimizer
can embed filter conditions and column selections directly into the
scan, reducing I/O at the source.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from titanframe.core.schema import Schema
from titanframe.expr.base import Expr
from titanframe.plan.logical.node import LogicalPlan


class ScanFormat(Enum):
    """Supported data source formats."""
    CSV = "csv"
    PARQUET = "parquet"
    ARROW_IPC = "arrow_ipc"
    JSON = "json"
    NDJSON = "ndjson"
    DATABASE = "database"
    IN_MEMORY = "in_memory"
    SQL = "sql"


class Scan(LogicalPlan):
    """
    Read data from an external source.

    Attributes:
        source: Path to the data source (file path or connection string).
        format: Data format (CSV, Parquet, etc.).
        schema: The full schema of the data source.
        projection: Optional list of column names to read (projection pushdown).
        predicate: Optional filter expression pushed down to the scan.
        limit: Optional row limit pushed down to the scan.
        chunk_size: Number of rows per chunk when reading.
        table: Optional table name for database sources.
    """

    __slots__ = (
        "source", "format", "_schema", "projection",
        "predicate", "limit", "chunk_size", "table",
    )

    def __init__(
        self,
        source: str | Path,
        format: ScanFormat,
        schema: Schema,
        projection: Optional[Sequence[str]] = None,
        predicate: Optional[Expr] = None,
        limit: Optional[int] = None,
        chunk_size: int = 65536,
        table: Optional[str] = None,
    ):
        self.source = str(source)
        self.format = format
        self._schema = schema
        self.projection = list(projection) if projection else None
        self.predicate = predicate
        self.limit = limit
        self.chunk_size = chunk_size
        self.table = table

    def output_schema(self) -> Schema:
        if self.projection:
            return self._schema.select(self.projection)
        return self._schema

    def children(self) -> list[LogicalPlan]:
        return []  # Scan is always a leaf

    def node_name(self) -> str:
        return "Scan"

    def node_description(self) -> str:
        parts = [f"source={self.source!r}", f"format={self.format.value}"]
        if self.projection:
            parts.append(f"projection={self.projection}")
        if self.predicate:
            parts.append(f"predicate={self.predicate}")
        if self.limit:
            parts.append(f"limit={self.limit}")
        return ", ".join(parts)

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Scan(
            self.source, self.format, self._schema,
            self.projection, self.predicate, self.limit, self.chunk_size,
        )

    def with_projection(self, columns: Sequence[str]) -> Scan:
        """Return a new Scan with projection pushdown applied."""
        return Scan(
            self.source, self.format, self._schema,
            list(columns), self.predicate, self.limit, self.chunk_size,
        )

    def with_predicate(self, predicate: Expr) -> Scan:
        """Return a new Scan with a predicate pushed down."""
        return Scan(
            self.source, self.format, self._schema,
            self.projection, predicate, self.limit, self.chunk_size,
        )

    def with_limit(self, limit: int) -> Scan:
        """Return a new Scan with a limit pushed down."""
        effective_limit = limit
        if self.limit is not None:
            effective_limit = min(self.limit, limit)
        return Scan(
            self.source, self.format, self._schema,
            self.projection, self.predicate, effective_limit, self.chunk_size,
        )

    @property
    def full_schema(self) -> Schema:
        """The full schema before any projection."""
        return self._schema
