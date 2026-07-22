"""
TitanFrame GroupBy
===================

GroupBy is a proxy object returned by ``DataFrame.group_by()``. It holds
the group keys and produces an ``Aggregation`` logical plan node when
``.agg()`` is called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from titanframe.expr.base import Expr
from titanframe.expr.column_expr import col, ColumnExpr

if TYPE_CHECKING:
    from titanframe.api.dataframe import DataFrame


class GroupBy:
    __slots__ = ("_df", "_keys")

    def __init__(self, df: DataFrame, keys: Sequence[str | Expr]):
        self._df = df
        self._keys: list[Expr] = [
            col(k) if isinstance(k, str) else k for k in keys
        ]

    def agg(self, *agg_exprs: Expr) -> DataFrame:
        from titanframe.api.dataframe import DataFrame
        lf = self._df.to_lazy()
        return lf.group_by(*self._keys).agg(*agg_exprs).collect()

    def sum(self, *columns: str) -> DataFrame:
        agg_exprs = [col(c).sum().alias(f"{c}_sum") for c in columns]
        return self.agg(*agg_exprs)

    def mean(self, *columns: str) -> DataFrame:
        agg_exprs = [col(c).mean().alias(f"{c}_mean") for c in columns]
        return self.agg(*agg_exprs)

    def count(self) -> DataFrame:
        first_key = self._keys[0]
        key_name = first_key.column_name if isinstance(first_key, ColumnExpr) else "count"
        return self.agg(col(key_name).count().alias("count"))

    def min(self, *columns: str) -> DataFrame:
        agg_exprs = [col(c).min().alias(f"{c}_min") for c in columns]
        return self.agg(*agg_exprs)

    def max(self, *columns: str) -> DataFrame:
        agg_exprs = [col(c).max().alias(f"{c}_max") for c in columns]
        return self.agg(*agg_exprs)

    def first(self, *columns: str) -> DataFrame:
        agg_exprs = [col(c).first().alias(c) for c in columns]
        return self.agg(*agg_exprs)

    def __repr__(self) -> str:
        key_names = [
            k.column_name if isinstance(k, ColumnExpr) else repr(k)
            for k in self._keys
        ]
        return f"GroupBy(keys={key_names})"


class LazyGroupBy:
    __slots__ = ("_lf", "_keys")

    def __init__(self, lf: Any, keys: Sequence[str | Expr]):
        self._lf = lf
        self._keys: list[Expr] = [
            col(k) if isinstance(k, str) else k for k in keys
        ]

    def agg(self, *agg_exprs: Expr) -> Any:
        from titanframe.api.lazyframe import LazyFrame
        from titanframe.plan.logical.aggregation import Aggregation

        agg_node = Aggregation(self._lf._plan, self._keys, list(agg_exprs))
        return LazyFrame(agg_node)

    def sum(self, *columns: str) -> Any:
        agg_exprs = [col(c).sum().alias(f"{c}_sum") for c in columns]
        return self.agg(*agg_exprs)

    def mean(self, *columns: str) -> Any:
        agg_exprs = [col(c).mean().alias(f"{c}_mean") for c in columns]
        return self.agg(*agg_exprs)

    def count(self) -> Any:
        first_key = self._keys[0]
        key_name = first_key.column_name if isinstance(first_key, ColumnExpr) else "count"
        return self.agg(col(key_name).count().alias("count"))

    def min(self, *columns: str) -> Any:
        agg_exprs = [col(c).min().alias(f"{c}_min") for c in columns]
        return self.agg(*agg_exprs)

    def max(self, *columns: str) -> Any:
        agg_exprs = [col(c).max().alias(f"{c}_max") for c in columns]
        return self.agg(*agg_exprs)

    def __repr__(self) -> str:
        key_names = [
            k.column_name if isinstance(k, ColumnExpr) else repr(k)
            for k in self._keys
        ]
        return f"LazyGroupBy(keys={key_names})"
