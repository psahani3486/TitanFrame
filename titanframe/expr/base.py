"""
TitanFrame Expression Base
==========================

The Expr class is the foundation of TitanFrame's lazy execution engine.
Every operation the user writes (``col("a") + 1``, ``col("b").sum()``)
creates an expression tree node — a piece of the computation DAG that
describes *what* to compute, not *how*.

This is the compiler's IR (Intermediate Representation).

Key design:
    - All Expr subclasses are **immutable** and **hashable**.
    - Python operator overloads (``__add__``, ``__gt__``, etc.) build the tree.
    - No data flows through expressions — they are purely symbolic.
    - The expression tree is evaluated by the physical engine's ExprEvaluator.

Example::

    >>> from titanframe.expr.column_expr import col
    >>> from titanframe.expr.literal_expr import lit
    >>> expr = (col("revenue") * lit(1.1)) > col("threshold")
    >>> print(expr)
    BinaryExpr(Op.GT, BinaryExpr(Op.MUL, ColumnExpr('revenue'), LiteralExpr(1.1)), ColumnExpr('threshold'))
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Sequence

from titanframe.core.dtypes import DType

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class Op(enum.Enum):
    """Binary operators."""
    # Arithmetic
    ADD = "+"
    SUB = "-"
    MUL = "*"
    TRUE_DIV = "/"
    FLOOR_DIV = "//"
    MOD = "%"
    POW = "**"

    # Comparison
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="

    # Logical
    AND = "&"
    OR = "|"
    XOR = "^"


class UnaryOp(enum.Enum):
    """Unary operators."""
    NEG = "-"
    NOT = "~"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    ABS = "abs"
    CEIL = "ceil"
    FLOOR = "floor"
    SQRT = "sqrt"
    LOG = "log"
    EXP = "exp"


class AggOp(enum.Enum):
    """Aggregation operators."""
    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    FIRST = "first"
    LAST = "last"
    STD = "std"
    VAR = "var"
    MEDIAN = "median"
    QUANTILE = "quantile"
    ANY = "any"
    ALL = "all"


class SortOrder(enum.Enum):
    """Sort direction."""
    ASC = "asc"
    DESC = "desc"


# ---------------------------------------------------------------------------
# Expr — abstract base
# ---------------------------------------------------------------------------

class Expr(ABC):
    """
    Abstract base class for all expression nodes.

    Subclasses represent different kinds of computations:
    ``ColumnExpr``, ``LiteralExpr``, ``BinaryExpr``, ``UnaryExpr``,
    ``AggExpr``, ``CastExpr``, ``AliasExpr``, etc.
    """

    __slots__ = ()

    @abstractmethod
    def children(self) -> list[Expr]:
        """Return child expressions (for tree traversal)."""
        ...

    @abstractmethod
    def display(self, indent: int = 0) -> str:
        """Pretty-print this expression tree."""
        ...

    def __repr__(self) -> str:
        return self.display()

    # ---- Column references used by this expression ----

    def required_columns(self) -> set[str]:
        """
        Recursively collect all column names referenced by this expression.

        Used by projection pushdown to determine which columns are needed.
        """
        cols: set[str] = set()
        self._collect_columns(cols)
        return cols

    def _collect_columns(self, out: set[str]) -> None:
        """Override in ColumnExpr to add its name."""
        for child in self.children():
            child._collect_columns(out)

    # ---- Tree walking ----

    def walk(self) -> list[Expr]:
        """Return all nodes in this expression tree (pre-order DFS)."""
        result: list[Expr] = [self]
        for child in self.children():
            result.extend(child.walk())
        return result

    def transform(self, fn: Any) -> Expr:
        """
        Apply ``fn`` to every node in the tree (bottom-up).

        ``fn`` receives each Expr and should return a (possibly new) Expr.
        """
        new_children = [child.transform(fn) for child in self.children()]
        new_self = self._with_children(new_children)
        return fn(new_self)

    @abstractmethod
    def _with_children(self, new_children: list[Expr]) -> Expr:
        """Return a copy of this node with replaced children."""
        ...

    # ---- Python operator overloads ----
    # These let users write ``col("a") + 1`` naturally.

    def __add__(self, other: Any) -> Expr:
        return BinaryExpr(Op.ADD, self, _wrap(other))

    def __radd__(self, other: Any) -> Expr:
        return BinaryExpr(Op.ADD, _wrap(other), self)

    def __sub__(self, other: Any) -> Expr:
        return BinaryExpr(Op.SUB, self, _wrap(other))

    def __rsub__(self, other: Any) -> Expr:
        return BinaryExpr(Op.SUB, _wrap(other), self)

    def __mul__(self, other: Any) -> Expr:
        return BinaryExpr(Op.MUL, self, _wrap(other))

    def __rmul__(self, other: Any) -> Expr:
        return BinaryExpr(Op.MUL, _wrap(other), self)

    def __truediv__(self, other: Any) -> Expr:
        return BinaryExpr(Op.TRUE_DIV, self, _wrap(other))

    def __rtruediv__(self, other: Any) -> Expr:
        return BinaryExpr(Op.TRUE_DIV, _wrap(other), self)

    def __floordiv__(self, other: Any) -> Expr:
        return BinaryExpr(Op.FLOOR_DIV, self, _wrap(other))

    def __rfloordiv__(self, other: Any) -> Expr:
        return BinaryExpr(Op.FLOOR_DIV, _wrap(other), self)

    def __mod__(self, other: Any) -> Expr:
        return BinaryExpr(Op.MOD, self, _wrap(other))

    def __rmod__(self, other: Any) -> Expr:
        return BinaryExpr(Op.MOD, _wrap(other), self)

    def __pow__(self, other: Any) -> Expr:
        return BinaryExpr(Op.POW, self, _wrap(other))

    def __rpow__(self, other: Any) -> Expr:
        return BinaryExpr(Op.POW, _wrap(other), self)

    # Comparison
    def __eq__(self, other: Any) -> Expr:  # type: ignore[override]
        return BinaryExpr(Op.EQ, self, _wrap(other))

    def __ne__(self, other: Any) -> Expr:  # type: ignore[override]
        return BinaryExpr(Op.NE, self, _wrap(other))

    def __lt__(self, other: Any) -> Expr:
        return BinaryExpr(Op.LT, self, _wrap(other))

    def __le__(self, other: Any) -> Expr:
        return BinaryExpr(Op.LE, self, _wrap(other))

    def __gt__(self, other: Any) -> Expr:
        return BinaryExpr(Op.GT, self, _wrap(other))

    def __ge__(self, other: Any) -> Expr:
        return BinaryExpr(Op.GE, self, _wrap(other))

    # Logical
    def __and__(self, other: Any) -> Expr:
        return BinaryExpr(Op.AND, self, _wrap(other))

    def __rand__(self, other: Any) -> Expr:
        return BinaryExpr(Op.AND, _wrap(other), self)

    def __or__(self, other: Any) -> Expr:
        return BinaryExpr(Op.OR, self, _wrap(other))

    def __ror__(self, other: Any) -> Expr:
        return BinaryExpr(Op.OR, _wrap(other), self)

    def __xor__(self, other: Any) -> Expr:
        return BinaryExpr(Op.XOR, self, _wrap(other))

    def __rxor__(self, other: Any) -> Expr:
        return BinaryExpr(Op.XOR, _wrap(other), self)

    # Unary
    def __neg__(self) -> Expr:
        return UnaryExpr(UnaryOp.NEG, self)

    def __invert__(self) -> Expr:
        return UnaryExpr(UnaryOp.NOT, self)

    def __abs__(self) -> Expr:
        return UnaryExpr(UnaryOp.ABS, self)

    # ---- Convenience methods ----

    def alias(self, name: str) -> Expr:
        """Give this expression an output name."""
        return AliasExpr(self, name)

    def cast(self, dtype: DType) -> Expr:
        """Cast this expression to a different type."""
        return CastExpr(self, dtype)

    def is_null(self) -> Expr:
        """Check if values are null."""
        return UnaryExpr(UnaryOp.IS_NULL, self)

    def is_not_null(self) -> Expr:
        """Check if values are not null."""
        return UnaryExpr(UnaryOp.IS_NOT_NULL, self)

    def abs(self) -> Expr:
        return UnaryExpr(UnaryOp.ABS, self)

    def sqrt(self) -> Expr:
        return UnaryExpr(UnaryOp.SQRT, self)

    def log(self) -> Expr:
        return UnaryExpr(UnaryOp.LOG, self)

    def exp(self) -> Expr:
        return UnaryExpr(UnaryOp.EXP, self)

    def ceil(self) -> Expr:
        return UnaryExpr(UnaryOp.CEIL, self)

    def floor(self) -> Expr:
        return UnaryExpr(UnaryOp.FLOOR, self)

    # ---- Aggregation methods ----

    def sum(self) -> Expr:
        """Sum aggregation."""
        return AggExpr(AggOp.SUM, self)

    def mean(self) -> Expr:
        """Mean aggregation."""
        return AggExpr(AggOp.MEAN, self)

    def min(self) -> Expr:
        """Min aggregation."""
        return AggExpr(AggOp.MIN, self)

    def max(self) -> Expr:
        """Max aggregation."""
        return AggExpr(AggOp.MAX, self)

    def count(self) -> Expr:
        """Count non-null values."""
        return AggExpr(AggOp.COUNT, self)

    def count_distinct(self) -> Expr:
        """Count distinct values."""
        return AggExpr(AggOp.COUNT_DISTINCT, self)

    def first(self) -> Expr:
        """First value."""
        return AggExpr(AggOp.FIRST, self)

    def last(self) -> Expr:
        """Last value."""
        return AggExpr(AggOp.LAST, self)

    def std(self) -> Expr:
        """Standard deviation."""
        return AggExpr(AggOp.STD, self)

    def var(self) -> Expr:
        """Variance."""
        return AggExpr(AggOp.VAR, self)

    def median(self) -> Expr:
        """Median value."""
        return AggExpr(AggOp.MEDIAN, self)

    # ---- String accessor ----

    @property
    def str(self) -> Any:
        """
        Access string operations via the ``.str`` namespace.

        Example::

            >>> col("name").str.lower()
            >>> col("email").str.contains("@gmail")
        """
        from titanframe.expr.string_expr import StringAccessor
        return StringAccessor(self)

    # ---- Datetime accessor ----

    @property
    def dt(self) -> Any:
        """
        Access datetime operations via the ``.dt`` namespace.

        Example::

            >>> col("created_at").dt.year
            >>> col("timestamp").dt.hour
        """
        from titanframe.expr.datetime_expr import DatetimeAccessor
        return DatetimeAccessor(self)

    # ---- Window accessor ----

    @property
    def window(self) -> Any:
        """
        Access window functions via the ``.window`` namespace.

        Example::

            >>> col("id").window.row_number().over("group")
        """
        from titanframe.expr.window_expr import WindowAccessor
        return WindowAccessor(self)

    def over(self, *partition_by: Any) -> Any:
        """
        Apply this expression as a window aggregation over partition keys.

        Shortcut for creating a WindowExpr with this as the aggregate.

        Example::

            >>> col("revenue").sum().over("region")
        """
        from titanframe.expr.window_expr import WindowExpr, DEFAULT_FRAME
        from titanframe.expr.column_expr import col as _col

        parts = [_col(p) if isinstance(p, str) else p for p in partition_by]

        # Determine if this is already an agg expression
        if isinstance(self, AggExpr):
            return WindowExpr(
                self.child, partition_by=parts, agg_op=self.op,
            )
        return WindowExpr(self, partition_by=parts)

    # ---- UDF methods ----

    def map(self, func: Any, return_dtype: Optional[DType] = None) -> Expr:
        """
        Apply a scalar Python function element-wise.

        Example::

            >>> col("name").map(lambda x: x.upper() if x else x)
        """
        from titanframe.expr.udf_expr import UDFExpr, UDFType
        return UDFExpr(self, func, UDFType.SCALAR, return_dtype)

    def map_batches(self, func: Any, return_dtype: Optional[DType] = None) -> Expr:
        """
        Apply a vectorized function to Arrow arrays.

        Example::

            >>> col("value").map_batches(lambda arr: pa.compute.add(arr, 1))
        """
        from titanframe.expr.udf_expr import UDFExpr, UDFType
        return UDFExpr(self, func, UDFType.VECTORIZED, return_dtype)

    # ---- Sort ----

    def asc(self) -> SortExpr:
        """Sort ascending."""
        return SortExpr(self, SortOrder.ASC)

    def desc(self) -> SortExpr:
        """Sort descending."""
        return SortExpr(self, SortOrder.DESC)

    # ---- Hash (for CSE detection) ----

    @abstractmethod
    def __hash__(self) -> int:
        ...


# ---------------------------------------------------------------------------
# Concrete expression nodes
# ---------------------------------------------------------------------------

class BinaryExpr(Expr):
    """Binary operation: ``left OP right``."""

    __slots__ = ("op", "left", "right")

    def __init__(self, op: Op, left: Expr, right: Expr):
        self.op = op
        self.left = left
        self.right = right

    def children(self) -> list[Expr]:
        return [self.left, self.right]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return BinaryExpr(self.op, new_children[0], new_children[1])

    def display(self, indent: int = 0) -> str:
        return f"BinaryExpr({self.op.name}, {self.left.display()}, {self.right.display()})"

    def __hash__(self) -> int:
        return hash(("binary", self.op, self.left, self.right))


class UnaryExpr(Expr):
    """Unary operation: ``OP(operand)``."""

    __slots__ = ("op", "operand")

    def __init__(self, op: UnaryOp, operand: Expr):
        self.op = op
        self.operand = operand

    def children(self) -> list[Expr]:
        return [self.operand]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return UnaryExpr(self.op, new_children[0])

    def display(self, indent: int = 0) -> str:
        return f"UnaryExpr({self.op.name}, {self.operand.display()})"

    def __hash__(self) -> int:
        return hash(("unary", self.op, self.operand))


class AggExpr(Expr):
    """Aggregation operation: ``AGG(child)``."""

    __slots__ = ("op", "child")

    def __init__(self, op: AggOp, child: Expr):
        self.op = op
        self.child = child

    def children(self) -> list[Expr]:
        return [self.child]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return AggExpr(self.op, new_children[0])

    def display(self, indent: int = 0) -> str:
        return f"AggExpr({self.op.name}, {self.child.display()})"

    def __hash__(self) -> int:
        return hash(("agg", self.op, self.child))


class CastExpr(Expr):
    """Type cast: ``CAST(child AS dtype)``."""

    __slots__ = ("child", "target_dtype")

    def __init__(self, child: Expr, target_dtype: DType):
        self.child = child
        self.target_dtype = target_dtype

    def children(self) -> list[Expr]:
        return [self.child]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return CastExpr(new_children[0], self.target_dtype)

    def display(self, indent: int = 0) -> str:
        return f"CastExpr({self.child.display()}, {self.target_dtype})"

    def __hash__(self) -> int:
        return hash(("cast", self.child, self.target_dtype))


class AliasExpr(Expr):
    """Rename an expression's output: ``expr.alias("new_name")``."""

    __slots__ = ("child", "name")

    def __init__(self, child: Expr, name: str):
        self.child = child
        self.name = name

    def children(self) -> list[Expr]:
        return [self.child]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return AliasExpr(new_children[0], self.name)

    def display(self, indent: int = 0) -> str:
        return f"AliasExpr({self.child.display()}, {self.name!r})"

    def __hash__(self) -> int:
        return hash(("alias", self.child, self.name))


class SortExpr(Expr):
    """Sort specification: ``expr.asc()`` or ``expr.desc()``."""

    __slots__ = ("child", "order")

    def __init__(self, child: Expr, order: SortOrder):
        self.child = child
        self.order = order

    def children(self) -> list[Expr]:
        return [self.child]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return SortExpr(new_children[0], self.order)

    def display(self, indent: int = 0) -> str:
        return f"SortExpr({self.child.display()}, {self.order.name})"

    def __hash__(self) -> int:
        return hash(("sort", self.child, self.order))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _wrap(value: Any) -> Expr:
    """
    Wrap a Python literal into a LiteralExpr if it isn't already an Expr.

    This enables natural syntax like ``col("a") + 1`` where ``1`` becomes
    ``LiteralExpr(1, Int64)``.
    """
    if isinstance(value, Expr):
        return value
    # Import here to avoid circular dependency
    from titanframe.expr.literal_expr import lit
    return lit(value)
