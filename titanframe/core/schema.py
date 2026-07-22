from __future__ import annotations
from collections import OrderedDict
from typing import Any, Iterator, Mapping, Sequence
import pyarrow as pa
from titanframe.core.dtypes import DType, from_arrow

class SchemaError(Exception):
    pass

class Schema:
    __slots__ = ('_fields',)

    def __init__(self, fields: Mapping[str, DType] | Sequence[tuple[str, DType]] | None=None):
        if fields is None:
            self._fields: OrderedDict[str, DType] = OrderedDict()
        elif isinstance(fields, Mapping):
            self._fields = OrderedDict(fields)
        else:
            self._fields = OrderedDict(fields)

    @property
    def names(self) -> list[str]:
        return list(self._fields.keys())

    @property
    def dtypes(self) -> list[DType]:
        return list(self._fields.values())

    @property
    def num_columns(self) -> int:
        return len(self._fields)

    def __getitem__(self, name: str) -> DType:
        if name not in self._fields:
            raise SchemaError(f'Column {name!r} not found in schema. Available columns: {self.names}')
        return self._fields[name]

    def __contains__(self, name: str) -> bool:
        return name in self._fields

    def __len__(self) -> int:
        return len(self._fields)

    def items(self) -> Iterator[tuple[str, DType]]:
        return iter(self._fields.items())

    def to_dict(self) -> dict[str, DType]:
        return dict(self._fields)

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Schema):
            return NotImplemented
        return self._fields == other._fields

    def __hash__(self) -> int:
        return hash(tuple(self._fields.items()))

    def __repr__(self) -> str:
        fields_str = ', '.join((f'{name!r}: {dtype}' for name, dtype in self._fields.items()))
        return f'Schema({{{fields_str}}})'

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise SchemaError(f'Column {name!r} not found in schema. Available: {self.names}')

    def field(self, index: int) -> tuple[str, DType]:
        if index < 0 or index >= len(self._fields):
            raise IndexError(f'Schema index {index} out of range [0, {len(self._fields)})')
        name = self.names[index]
        return (name, self._fields[name])

    def select(self, names: Sequence[str]) -> Schema:
        fields = OrderedDict()
        for name in names:
            if name not in self._fields:
                raise SchemaError(f'Cannot select column {name!r}: not in schema. Available: {self.names}')
            fields[name] = self._fields[name]
        return Schema(fields)

    def drop(self, names: Sequence[str]) -> Schema:
        drop_set = set(names)
        fields = OrderedDict(((name, dtype) for name, dtype in self._fields.items() if name not in drop_set))
        return Schema(fields)

    def rename(self, mapping: Mapping[str, str]) -> Schema:
        fields = OrderedDict()
        for name, dtype in self._fields.items():
            new_name = mapping.get(name, name)
            if new_name in fields:
                raise SchemaError(f'Rename collision: both {name!r} and another column map to {new_name!r}')
            fields[new_name] = dtype
        return Schema(fields)

    def merge(self, other: Schema, suffix: str='_right') -> Schema:
        fields = OrderedDict(self._fields)
        for name, dtype in other._fields.items():
            final_name = name
            if final_name in fields:
                final_name = f'{name}{suffix}'
                if final_name in fields:
                    raise SchemaError(f'Merge collision even with suffix: {final_name!r} already exists')
            fields[final_name] = dtype
        return Schema(fields)

    def intersect(self, other: Schema) -> Schema:
        fields = OrderedDict()
        for name, dtype in self._fields.items():
            if name in other._fields and other._fields[name] == dtype:
                fields[name] = dtype
        return Schema(fields)

    def diff(self, other: Schema) -> Schema:
        fields = OrderedDict()
        for name, dtype in self._fields.items():
            if name not in other._fields:
                fields[name] = dtype
        return Schema(fields)

    def append(self, name: str, dtype: DType) -> Schema:
        if name in self._fields:
            raise SchemaError(f'Column {name!r} already exists in schema')
        fields = OrderedDict(self._fields)
        fields[name] = dtype
        return Schema(fields)

    def with_column(self, name: str, dtype: DType) -> Schema:
        fields = OrderedDict(self._fields)
        fields[name] = dtype
        return Schema(fields)

    def is_compatible(self, other: Schema) -> bool:
        return self == other

    def is_subset_of(self, other: Schema) -> bool:
        for name, dtype in self._fields.items():
            if name not in other._fields or other._fields[name] != dtype:
                return False
        return True

    def assert_compatible(self, other: Schema, context: str='') -> None:
        if self == other:
            return
        prefix = f' during {context}' if context else ''
        if len(self) != len(other):
            raise SchemaError(f'Schema mismatch{prefix}: left has {len(self)} columns, right has {len(other)} columns')
        for (l_name, l_dtype), (r_name, r_dtype) in zip(self._fields.items(), other._fields.items()):
            if l_name != r_name:
                raise SchemaError(f'Schema mismatch{prefix}: column name {l_name!r} (left) != {r_name!r} (right)')
            if l_dtype != r_dtype:
                raise SchemaError(f'Schema mismatch{prefix}: column {l_name!r} type {l_dtype} (left) != {r_dtype} (right)')

    def to_arrow(self) -> pa.Schema:
        fields = [pa.field(name, dtype.arrow_type) for name, dtype in self._fields.items()]
        return pa.schema(fields)

    @classmethod
    def from_arrow(cls, arrow_schema: pa.Schema) -> Schema:
        fields = OrderedDict()
        for i in range(len(arrow_schema)):
            field = arrow_schema.field(i)
            fields[field.name] = from_arrow(field.type)
        return cls(fields)

    @classmethod
    def from_dict(cls, data: dict[str, list[Any]]) -> Schema:
        from titanframe.core.dtypes import from_value, Null
        fields = OrderedDict()
        for name, values in data.items():
            dtype = Null
            for v in values:
                if v is not None:
                    dtype = from_value(v)
                    break
            fields[name] = dtype
        return cls(fields)
