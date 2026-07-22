"""
TitanFrame Schema
=================

A Schema is an ordered mapping of column names to DTypes. It is the structural
contract of every Table, Chunk, and DataFrame — all data flowing through the
engine is validated against a Schema.

Schemas support algebraic operations (merge, intersect, diff, rename) that
enable the query optimizer to propagate type information through the plan DAG.

Example::

    >>> from titanframe.core.schema import Schema
    >>> from titanframe.core.dtypes import Int64, Utf8, Float64
    >>> s = Schema({"id": Int64, "name": Utf8, "revenue": Float64})
    >>> s.names
    ['id', 'name', 'revenue']
    >>> s.select(["id", "revenue"])
    Schema({'id': Int64, 'revenue': Float64})
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterator, Mapping, Sequence

import pyarrow as pa

from titanframe.core.dtypes import DType, from_arrow


class SchemaError(Exception):
    """Raised when schema operations fail (incompatible schemas, missing columns, etc.)."""
    pass


class Schema:
    """
    An ordered mapping of column names to :class:`DType`.

    The schema is immutable after construction — all mutation methods return
    new Schema instances.

    Args:
        fields: Mapping of column name → DType, or a list of (name, DType) tuples.
    """

    __slots__ = ("_fields",)

    def __init__(self, fields: Mapping[str, DType] | Sequence[tuple[str, DType]] | None = None):
        if fields is None:
            self._fields: OrderedDict[str, DType] = OrderedDict()
        elif isinstance(fields, Mapping):
            self._fields = OrderedDict(fields)
        else:
            self._fields = OrderedDict(fields)

    # ---- Properties ----

    @property
    def names(self) -> list[str]:
        """Column names in order."""
        return list(self._fields.keys())

    @property
    def dtypes(self) -> list[DType]:
        """Column types in order."""
        return list(self._fields.values())

    @property
    def num_columns(self) -> int:
        """Number of columns."""
        return len(self._fields)

    # ---- Lookups ----

    def __getitem__(self, name: str) -> DType:
        """Get the DType for a column by name."""
        if name not in self._fields:
            raise SchemaError(
                f"Column {name!r} not found in schema. "
                f"Available columns: {self.names}"
            )
        return self._fields[name]

    def __contains__(self, name: str) -> bool:
        """Check if a column name exists in the schema."""
        return name in self._fields

    def __len__(self) -> int:
        return len(self._fields)

    def items(self) -> Iterator[tuple[str, DType]]:
        """Iterator over (name, dtype) pairs."""
        return iter(self._fields.items())

    def to_dict(self) -> dict[str, DType]:
        """Convert schema to dictionary mapping column name to DType."""
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
        fields_str = ", ".join(f"{name!r}: {dtype}" for name, dtype in self._fields.items())
        return f"Schema({{{fields_str}}})"

    def index(self, name: str) -> int:
        """
        Get the positional index of a column.

        Raises:
            SchemaError: If the column is not found.
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise SchemaError(
                f"Column {name!r} not found in schema. Available: {self.names}"
            )

    def field(self, index: int) -> tuple[str, DType]:
        """Get the (name, dtype) pair at a positional index."""
        if index < 0 or index >= len(self._fields):
            raise IndexError(f"Schema index {index} out of range [0, {len(self._fields)})")
        name = self.names[index]
        return name, self._fields[name]

    # ---- Algebra ----

    def select(self, names: Sequence[str]) -> Schema:
        """
        Return a new schema with only the specified columns, preserving order.

        Raises:
            SchemaError: If any name is not in the schema.
        """
        fields = OrderedDict()
        for name in names:
            if name not in self._fields:
                raise SchemaError(
                    f"Cannot select column {name!r}: not in schema. Available: {self.names}"
                )
            fields[name] = self._fields[name]
        return Schema(fields)

    def drop(self, names: Sequence[str]) -> Schema:
        """Return a new schema without the specified columns."""
        drop_set = set(names)
        fields = OrderedDict(
            (name, dtype) for name, dtype in self._fields.items()
            if name not in drop_set
        )
        return Schema(fields)

    def rename(self, mapping: Mapping[str, str]) -> Schema:
        """
        Return a new schema with columns renamed according to ``mapping``.

        Only keys present in the mapping are renamed; others keep their names.
        """
        fields = OrderedDict()
        for name, dtype in self._fields.items():
            new_name = mapping.get(name, name)
            if new_name in fields:
                raise SchemaError(
                    f"Rename collision: both {name!r} and another column map to {new_name!r}"
                )
            fields[new_name] = dtype
        return Schema(fields)

    def merge(self, other: Schema, suffix: str = "_right") -> Schema:
        """
        Merge two schemas. On name collision, append ``suffix`` to the right schema's column.

        This is used for join operations.
        """
        fields = OrderedDict(self._fields)
        for name, dtype in other._fields.items():
            final_name = name
            if final_name in fields:
                final_name = f"{name}{suffix}"
                if final_name in fields:
                    raise SchemaError(
                        f"Merge collision even with suffix: {final_name!r} already exists"
                    )
            fields[final_name] = dtype
        return Schema(fields)

    def intersect(self, other: Schema) -> Schema:
        """Return a schema containing only columns present in both schemas (by name and type)."""
        fields = OrderedDict()
        for name, dtype in self._fields.items():
            if name in other._fields and other._fields[name] == dtype:
                fields[name] = dtype
        return Schema(fields)

    def diff(self, other: Schema) -> Schema:
        """Return columns in ``self`` but not in ``other``."""
        fields = OrderedDict()
        for name, dtype in self._fields.items():
            if name not in other._fields:
                fields[name] = dtype
        return Schema(fields)

    def append(self, name: str, dtype: DType) -> Schema:
        """Return a new schema with an additional column at the end."""
        if name in self._fields:
            raise SchemaError(f"Column {name!r} already exists in schema")
        fields = OrderedDict(self._fields)
        fields[name] = dtype
        return Schema(fields)

    def with_column(self, name: str, dtype: DType) -> Schema:
        """
        Return a schema with the column added or its type replaced.

        Unlike ``append``, this does not raise on duplicates — it replaces.
        """
        fields = OrderedDict(self._fields)
        fields[name] = dtype
        return Schema(fields)

    # ---- Validation ----

    def is_compatible(self, other: Schema) -> bool:
        """
        Check if ``other`` is structurally compatible with ``self``.

        Compatible means same column names in same order and same types.
        """
        return self == other

    def is_subset_of(self, other: Schema) -> bool:
        """Check if all columns in ``self`` exist in ``other`` with matching types."""
        for name, dtype in self._fields.items():
            if name not in other._fields or other._fields[name] != dtype:
                return False
        return True

    def assert_compatible(self, other: Schema, context: str = "") -> None:
        """
        Raise SchemaError if schemas are not compatible.

        Args:
            other: Schema to compare against.
            context: Optional context string for error messages (e.g., "union operation").
        """
        if self == other:
            return

        prefix = f" during {context}" if context else ""

        # Check column count
        if len(self) != len(other):
            raise SchemaError(
                f"Schema mismatch{prefix}: "
                f"left has {len(self)} columns, right has {len(other)} columns"
            )

        # Check each column
        for (l_name, l_dtype), (r_name, r_dtype) in zip(
            self._fields.items(), other._fields.items()
        ):
            if l_name != r_name:
                raise SchemaError(
                    f"Schema mismatch{prefix}: "
                    f"column name {l_name!r} (left) != {r_name!r} (right)"
                )
            if l_dtype != r_dtype:
                raise SchemaError(
                    f"Schema mismatch{prefix}: "
                    f"column {l_name!r} type {l_dtype} (left) != {r_dtype} (right)"
                )

    # ---- Arrow interop ----

    def to_arrow(self) -> pa.Schema:
        """Convert to a ``pyarrow.Schema``."""
        fields = [pa.field(name, dtype.arrow_type) for name, dtype in self._fields.items()]
        return pa.schema(fields)

    @classmethod
    def from_arrow(cls, arrow_schema: pa.Schema) -> Schema:
        """
        Construct a Schema from a ``pyarrow.Schema``.

        Example::

            >>> import pyarrow as pa
            >>> arrow_s = pa.schema([("a", pa.int32()), ("b", pa.utf8())])
            >>> Schema.from_arrow(arrow_s)
            Schema({'a': Int32, 'b': Utf8})
        """
        fields = OrderedDict()
        for i in range(len(arrow_schema)):
            field = arrow_schema.field(i)
            fields[field.name] = from_arrow(field.type)
        return cls(fields)

    # ---- Convenience constructors ----

    @classmethod
    def from_dict(cls, data: dict[str, list[Any]]) -> Schema:
        """
        Infer a schema from a Python dict of lists (like Pandas DataFrame constructor).

        Uses the first non-None value in each column to infer the type.
        """
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
