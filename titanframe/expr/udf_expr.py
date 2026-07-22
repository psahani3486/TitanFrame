"""
User-Defined Function Expression — ``col("x").map(fn)``
=========================================================

Wraps arbitrary Python callables as expression nodes, enabling custom
transformations within the lazy computation graph.

UDFs are the escape hatch — when no built-in expression covers the
user's need, they can provide a Python function that operates on
Arrow arrays or Python scalars.

Example::

    >>> from titanframe import col
    >>> # Scalar UDF (applied element-wise)
    >>> col("name").map(lambda x: x.upper() if x else x)
    >>> # Vectorized UDF (applied to Arrow arrays)
    >>> col("value").map_batches(lambda arr: pa.compute.add(arr, 1))
"""

from __future__ import annotations

import enum
from typing import Any, Callable, Optional

from titanframe.core.dtypes import DType
from titanframe.expr.base import Expr


class UDFType(enum.Enum):
    """Type of user-defined function."""
    SCALAR = "scalar"
    VECTORIZED = "vectorized"


class UDFExpr(Expr):
    """
    A user-defined function applied to an expression.

    Attributes:
        child: The input expression.
        func: The Python callable.
        udf_type: Whether the function operates on scalars or batches.
        return_dtype: The output data type. Must be specified for vectorized UDFs.
        func_name: Human-readable name for display/debugging.
    """

    __slots__ = ("child", "func", "udf_type", "return_dtype", "func_name")

    def __init__(
        self,
        child: Expr,
        func: Callable[..., Any],
        udf_type: UDFType = UDFType.SCALAR,
        return_dtype: Optional[DType] = None,
        func_name: Optional[str] = None,
    ):
        self.child = child
        self.func = func
        self.udf_type = udf_type
        self.return_dtype = return_dtype
        self.func_name = func_name or getattr(func, "__name__", "<lambda>")

    def children(self) -> list[Expr]:
        return [self.child]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return UDFExpr(
            new_children[0], self.func, self.udf_type,
            self.return_dtype, self.func_name,
        )

    def display(self, indent: int = 0) -> str:
        return (
            f"UDFExpr({self.udf_type.value}, func={self.func_name!r}, "
            f"child={self.child.display()}, return_dtype={self.return_dtype})"
        )

    def __hash__(self) -> int:
        return hash(("udf", id(self.func), self.child, self.udf_type))


class MultiColumnUDFExpr(Expr):
    """
    A UDF that takes multiple columns as input.

    Example::

        >>> # Compute distance from two coordinate columns
        >>> multi_udf(
        ...     [col("lat"), col("lon")],
        ...     func=lambda lat, lon: (lat**2 + lon**2)**0.5,
        ...     return_dtype=Float64,
        ... )
    """

    __slots__ = ("inputs", "func", "udf_type", "return_dtype", "func_name")

    def __init__(
        self,
        inputs: list[Expr],
        func: Callable[..., Any],
        udf_type: UDFType = UDFType.SCALAR,
        return_dtype: Optional[DType] = None,
        func_name: Optional[str] = None,
    ):
        self.inputs = list(inputs)
        self.func = func
        self.udf_type = udf_type
        self.return_dtype = return_dtype
        self.func_name = func_name or getattr(func, "__name__", "<lambda>")

    def children(self) -> list[Expr]:
        return list(self.inputs)

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return MultiColumnUDFExpr(
            new_children, self.func, self.udf_type,
            self.return_dtype, self.func_name,
        )

    def display(self, indent: int = 0) -> str:
        inputs_str = ", ".join(e.display() for e in self.inputs)
        return (
            f"MultiColumnUDFExpr({self.udf_type.value}, func={self.func_name!r}, "
            f"inputs=[{inputs_str}], return_dtype={self.return_dtype})"
        )

    def __hash__(self) -> int:
        return hash(("multi_udf", id(self.func), tuple(self.inputs), self.udf_type))
