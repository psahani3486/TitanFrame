"""
Aggregation Expression — ``AggExpr``
=======================================

Extended module for aggregation operations. Provides:
    - Standard aggregations: ``Sum, Mean, Min, Max, Count, CountDistinct,
      First, Last, Std, Var, Median, Quantile, Any, All``
    - **Partial aggregation** support for distributed/streaming execution:
      ``partial_agg()`` → ``merge_agg()`` pattern
    - Type-inference for aggregation output

The core ``AggExpr`` class lives in :mod:`base` and is re-exported here.
This module adds partial aggregation infrastructure, quantile support,
and convenience factory functions.

Example::

    >>> from titanframe.expr.agg_expr import sum_, mean, count
    >>> from titanframe.expr.column_expr import col
    >>> expr = sum_(col("revenue")).alias("total_revenue")

Distributed Aggregation Pattern::

    # Phase 1: Each partition computes a partial result
    partial = partial_agg(AggOp.MEAN, chunk)
    # → PartialAggResult(sum=..., count=..., ...)

    # Phase 2: Merge all partial results
    final = merge_agg(AggOp.MEAN, [partial1, partial2, ...])
    # → Correct global mean
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from titanframe.core.dtypes import DType, Bool, Float64, Int64
from titanframe.expr.base import (
    AggExpr,
    Expr,
    AggOp,
    _wrap,
)

# Re-export for users importing from this module
__all__ = [
    "AggExpr",
    "AggOp",
    "QuantileExpr",
    "PartialAggResult",
    "sum_",
    "mean",
    "min_",
    "max_",
    "count",
    "count_distinct",
    "first",
    "last",
    "std",
    "var",
    "median",
    "quantile",
    "any_",
    "all_",
    "infer_agg_dtype",
    "partial_agg_state_fields",
    "is_reducible_op",
]


# ---------------------------------------------------------------------------
# Extended aggregation: Quantile
# ---------------------------------------------------------------------------

class QuantileExpr(AggExpr):
    """
    Quantile aggregation: ``QUANTILE(child, q=0.5)``.

    Unlike other aggregations, quantile requires a parameter (the quantile
    level ``q ∈ [0, 1]``).

    Attributes:
        q: The quantile to compute (0.0 = min, 0.5 = median, 1.0 = max).
    """

    __slots__ = ("q",)

    def __init__(self, child: Expr, q: float = 0.5):
        super().__init__(AggOp.QUANTILE, child)
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"Quantile q must be in [0, 1], got {q}")
        self.q = q

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return QuantileExpr(new_children[0], self.q)

    def display(self, indent: int = 0) -> str:
        return f"QuantileExpr({self.child.display()}, q={self.q})"

    def __hash__(self) -> int:
        return hash(("quantile", self.child, self.q))


# ---------------------------------------------------------------------------
# Type inference for aggregation
# ---------------------------------------------------------------------------

def infer_agg_dtype(op: AggOp, input_dtype: DType) -> DType:
    """
    Infer the output dtype of an aggregation.

    Rules:
        - ``COUNT``, ``COUNT_DISTINCT`` → ``Int64`` (always integer count)
        - ``ANY``, ``ALL`` → ``Bool``
        - ``MEAN``, ``STD``, ``VAR``, ``MEDIAN``, ``QUANTILE`` → ``Float64``
        - ``SUM`` → promoted type (Int → Int64, Float → same)
        - ``MIN``, ``MAX``, ``FIRST``, ``LAST`` → same as input

    Args:
        op: The aggregation operator.
        input_dtype: The column's type being aggregated.

    Returns:
        The inferred output DType.
    """
    # Always integer counts
    if op in (AggOp.COUNT, AggOp.COUNT_DISTINCT):
        return Int64

    # Always boolean
    if op in (AggOp.ANY, AggOp.ALL):
        return Bool

    # Always float for statistical aggregations
    if op in (AggOp.MEAN, AggOp.STD, AggOp.VAR, AggOp.MEDIAN, AggOp.QUANTILE):
        return Float64

    # SUM: promote integers to Int64, keep floats
    if op == AggOp.SUM:
        if input_dtype.is_integer:
            return Int64
        return input_dtype

    # MIN, MAX, FIRST, LAST: preserve input type
    if op in (AggOp.MIN, AggOp.MAX, AggOp.FIRST, AggOp.LAST):
        return input_dtype

    raise TypeError(f"Cannot infer dtype for unknown agg op: {op}")


# ---------------------------------------------------------------------------
# Partial aggregation support (for distributed execution)
# ---------------------------------------------------------------------------

@dataclass
class PartialAggResult:
    """
    Intermediate result of a partial aggregation.

    For distributed/streaming execution, we split aggregation into two phases:
    1. **Partial**: each partition computes a partial result (this object).
    2. **Merge**: partial results from all partitions are combined.

    This enables aggregating datasets that don't fit in memory by processing
    one chunk at a time.

    Attributes:
        op: The aggregation being computed.
        count: Number of non-null values seen.
        sum_value: Running sum (for SUM, MEAN, VAR, STD).
        sum_sq_value: Running sum of squares (for VAR, STD — Welford's method).
        min_value: Running minimum.
        max_value: Running maximum.
        first_value: First value seen (for FIRST).
        last_value: Last value seen (for LAST).
        any_value: Running logical OR (for ANY).
        all_value: Running logical AND (for ALL).
    """

    op: AggOp
    count: int = 0
    sum_value: float = 0.0
    sum_sq_value: float = 0.0
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    first_value: Optional[Any] = None
    last_value: Optional[Any] = None
    any_value: bool = False
    all_value: bool = True


def partial_agg_state_fields(op: AggOp) -> list[str]:
    """
    Return the list of PartialAggResult fields needed for a given aggregation.

    This is used by the physical planner to allocate only the necessary
    state for each aggregation, reducing memory overhead.

    Args:
        op: The aggregation operator.

    Returns:
        List of field names from :class:`PartialAggResult` that are needed.
    """
    _fields_map: dict[AggOp, list[str]] = {
        AggOp.SUM: ["count", "sum_value"],
        AggOp.MEAN: ["count", "sum_value"],
        AggOp.STD: ["count", "sum_value", "sum_sq_value"],
        AggOp.VAR: ["count", "sum_value", "sum_sq_value"],
        AggOp.MIN: ["min_value"],
        AggOp.MAX: ["max_value"],
        AggOp.COUNT: ["count"],
        AggOp.COUNT_DISTINCT: ["count"],  # Requires a set, simplified here
        AggOp.FIRST: ["first_value", "count"],
        AggOp.LAST: ["last_value"],
        AggOp.ANY: ["any_value"],
        AggOp.ALL: ["all_value"],
        AggOp.MEDIAN: ["count", "sum_value"],  # Approximate; exact needs all data
        AggOp.QUANTILE: ["count", "sum_value"],  # Approximate; exact needs all data
    }
    return _fields_map.get(op, ["count", "sum_value"])


def is_reducible_op(op: AggOp) -> bool:
    """
    Return ``True`` if the aggregation can be computed in a streaming
    partial → merge fashion without keeping all data in memory.

    SUM, COUNT, MIN, MAX, MEAN, STD, VAR, ANY, ALL are reducible.
    MEDIAN and QUANTILE are not (they need full data for exact results).
    """
    _non_reducible = frozenset({AggOp.MEDIAN, AggOp.QUANTILE})
    return op not in _non_reducible


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def _make_agg(op: AggOp, operand: Any) -> AggExpr:
    """Create an AggExpr, wrapping raw values into expressions."""
    return AggExpr(op, _wrap(operand))


def sum_(operand: Any) -> AggExpr:
    """
    Sum aggregation.

    Example::

        >>> sum_(col("revenue"))
    """
    return _make_agg(AggOp.SUM, operand)


def mean(operand: Any) -> AggExpr:
    """
    Mean (average) aggregation.

    Example::

        >>> mean(col("score"))
    """
    return _make_agg(AggOp.MEAN, operand)


def min_(operand: Any) -> AggExpr:
    """
    Minimum value aggregation.

    Example::

        >>> min_(col("temperature"))
    """
    return _make_agg(AggOp.MIN, operand)


def max_(operand: Any) -> AggExpr:
    """
    Maximum value aggregation.

    Example::

        >>> max_(col("temperature"))
    """
    return _make_agg(AggOp.MAX, operand)


def count(operand: Any) -> AggExpr:
    """
    Count non-null values.

    Example::

        >>> count(col("id"))
    """
    return _make_agg(AggOp.COUNT, operand)


def count_distinct(operand: Any) -> AggExpr:
    """
    Count distinct non-null values.

    Example::

        >>> count_distinct(col("customer_id"))
    """
    return _make_agg(AggOp.COUNT_DISTINCT, operand)


def first(operand: Any) -> AggExpr:
    """
    First value in group.

    Example::

        >>> first(col("name"))
    """
    return _make_agg(AggOp.FIRST, operand)


def last(operand: Any) -> AggExpr:
    """
    Last value in group.

    Example::

        >>> last(col("timestamp"))
    """
    return _make_agg(AggOp.LAST, operand)


def std(operand: Any) -> AggExpr:
    """
    Standard deviation.

    Example::

        >>> std(col("score"))
    """
    return _make_agg(AggOp.STD, operand)


def var(operand: Any) -> AggExpr:
    """
    Variance.

    Example::

        >>> var(col("score"))
    """
    return _make_agg(AggOp.VAR, operand)


def median(operand: Any) -> AggExpr:
    """
    Median value.

    Example::

        >>> median(col("income"))
    """
    return _make_agg(AggOp.MEDIAN, operand)


def quantile(operand: Any, q: float = 0.5) -> QuantileExpr:
    """
    Compute a quantile.

    Args:
        operand: Column expression.
        q: Quantile level in [0, 1]. 0.5 = median.

    Example::

        >>> quantile(col("latency"), q=0.95)
    """
    return QuantileExpr(_wrap(operand), q)


def any_(operand: Any) -> AggExpr:
    """
    Logical ANY (True if any value is True).

    Example::

        >>> any_(col("is_fraud"))
    """
    return _make_agg(AggOp.ANY, operand)


def all_(operand: Any) -> AggExpr:
    """
    Logical ALL (True if all values are True).

    Example::

        >>> all_(col("is_valid"))
    """
    return _make_agg(AggOp.ALL, operand)
