"""
Sink Node — Write Targets
=========================
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from titanframe.core.schema import Schema
from titanframe.plan.logical.node import LogicalPlan


class SinkFormat(Enum):
    """Supported data target formats."""
    CSV = "csv"
    PARQUET = "parquet"
    ARROW_IPC = "arrow_ipc"
    JSON = "json"
    NDJSON = "ndjson"


class Sink(LogicalPlan):
    """
    Write data to an external target.
    
    This is typically the root of a logical plan that executes side effects.
    The output schema of a sink is usually empty, or the same as the input
    if returning the written rows is desired. We'll use the input schema.
    """
    __slots__ = ("input", "target", "format", "options")

    def __init__(
        self,
        input: LogicalPlan,
        target: str | Path,
        format: SinkFormat,
        **options: Any
    ):
        self.input = input
        self.target = str(target)
        self.format = format
        self.options = options

    def output_schema(self) -> Schema:
        return self.input.output_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_name(self) -> str:
        return "Sink"

    def node_description(self) -> str:
        return f"target='{self.target}', format={self.format.value}"

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Sink(new_children[0], self.target, self.format, **self.options)
