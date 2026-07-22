from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from titanframe.core.schema import Schema
from titanframe.plan.logical.node import LogicalPlan

class SinkFormat(Enum):
    CSV = 'csv'
    PARQUET = 'parquet'
    ARROW_IPC = 'arrow_ipc'
    JSON = 'json'
    NDJSON = 'ndjson'

class Sink(LogicalPlan):
    __slots__ = ('input', 'target', 'format', 'options')

    def __init__(self, input: LogicalPlan, target: str | Path, format: SinkFormat, **options: Any):
        self.input = input
        self.target = str(target)
        self.format = format
        self.options = options

    def output_schema(self) -> Schema:
        return self.input.output_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_name(self) -> str:
        return 'Sink'

    def node_description(self) -> str:
        return f"target='{self.target}', format={self.format.value}"

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Sink(new_children[0], self.target, self.format, **self.options)
