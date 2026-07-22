"""
Datetime Expression Accessor — ``col("ts").dt.year``
=====================================================

Provides datetime-specific operations as a namespace accessor,
mirroring Pandas' ``Series.dt`` accessor.

Example::

    >>> from titanframe import col
    >>> expr = col("created_at").dt.year
    >>> expr = col("timestamp").dt.hour
    >>> expr = col("date").dt.weekday()
"""

from __future__ import annotations

import enum
from typing import Optional

from titanframe.expr.base import Expr


class DatetimeOp(enum.Enum):
    """Datetime component extraction and manipulation operations."""
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"
    MILLISECOND = "millisecond"
    MICROSECOND = "microsecond"
    NANOSECOND = "nanosecond"
    WEEKDAY = "weekday"
    ISO_WEEKDAY = "iso_weekday"
    WEEK = "week"
    DAY_OF_YEAR = "day_of_year"
    QUARTER = "quarter"

    DATE = "date"
    TIME = "time"
    EPOCH = "epoch"
    EPOCH_MS = "epoch_ms"

    TRUNCATE = "truncate"
    ROUND = "round"
    CEIL = "ceil"

    OFFSET_BY = "offset_by"
    DIFF = "diff"

    STRFTIME = "strftime"
    TIMESTAMP = "timestamp"


class DatetimeExpr(Expr):
    """
    A datetime operation applied to an expression.

    Attributes:
        child: The input expression (must produce temporal values).
        op: The datetime operation to apply.
        args: Additional arguments (format string, unit, etc.).
    """

    __slots__ = ("child", "op", "args")

    def __init__(self, child: Expr, op: DatetimeOp, args: tuple = ()):
        self.child = child
        self.op = op
        self.args = args

    def children(self) -> list[Expr]:
        return [self.child]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return DatetimeExpr(new_children[0], self.op, self.args)

    def display(self, indent: int = 0) -> str:
        args_str = f", {self.args}" if self.args else ""
        return f"DatetimeExpr({self.op.value}, {self.child.display()}{args_str})"

    def __hash__(self) -> int:
        return hash(("datetime", self.child, self.op, self.args))


class DatetimeAccessor:
    """
    Namespace accessor for datetime operations on an expression.

    Accessed via ``expr.dt`` — mirrors the Pandas ``Series.dt`` API.

    Example::

        >>> col("created_at").dt.year
        >>> col("timestamp").dt.hour
        >>> col("date").dt.strftime("%Y-%m-%d")
    """

    __slots__ = ("_expr",)

    def __init__(self, expr: Expr):
        self._expr = expr


    @property
    def year(self) -> Expr:
        """Extract the year component."""
        return DatetimeExpr(self._expr, DatetimeOp.YEAR)

    @property
    def month(self) -> Expr:
        """Extract the month component (1–12)."""
        return DatetimeExpr(self._expr, DatetimeOp.MONTH)

    @property
    def day(self) -> Expr:
        """Extract the day component (1–31)."""
        return DatetimeExpr(self._expr, DatetimeOp.DAY)

    @property
    def hour(self) -> Expr:
        """Extract the hour component (0–23)."""
        return DatetimeExpr(self._expr, DatetimeOp.HOUR)

    @property
    def minute(self) -> Expr:
        """Extract the minute component (0–59)."""
        return DatetimeExpr(self._expr, DatetimeOp.MINUTE)

    @property
    def second(self) -> Expr:
        """Extract the second component (0–59)."""
        return DatetimeExpr(self._expr, DatetimeOp.SECOND)

    @property
    def millisecond(self) -> Expr:
        """Extract the millisecond component."""
        return DatetimeExpr(self._expr, DatetimeOp.MILLISECOND)

    @property
    def microsecond(self) -> Expr:
        """Extract the microsecond component."""
        return DatetimeExpr(self._expr, DatetimeOp.MICROSECOND)

    @property
    def nanosecond(self) -> Expr:
        """Extract the nanosecond component."""
        return DatetimeExpr(self._expr, DatetimeOp.NANOSECOND)

    @property
    def weekday(self) -> Expr:
        """Day of the week (Monday=0, Sunday=6)."""
        return DatetimeExpr(self._expr, DatetimeOp.WEEKDAY)

    @property
    def iso_weekday(self) -> Expr:
        """ISO day of the week (Monday=1, Sunday=7)."""
        return DatetimeExpr(self._expr, DatetimeOp.ISO_WEEKDAY)

    @property
    def week(self) -> Expr:
        """ISO week number (1–53)."""
        return DatetimeExpr(self._expr, DatetimeOp.WEEK)

    @property
    def day_of_year(self) -> Expr:
        """Day of the year (1–366)."""
        return DatetimeExpr(self._expr, DatetimeOp.DAY_OF_YEAR)

    @property
    def quarter(self) -> Expr:
        """Quarter of the year (1–4)."""
        return DatetimeExpr(self._expr, DatetimeOp.QUARTER)

    @property
    def date(self) -> Expr:
        """Extract the date part (removes time)."""
        return DatetimeExpr(self._expr, DatetimeOp.DATE)

    @property
    def time(self) -> Expr:
        """Extract the time part (removes date)."""
        return DatetimeExpr(self._expr, DatetimeOp.TIME)


    def epoch(self, unit: str = "s") -> Expr:
        """
        Convert to epoch time.

        Args:
            unit: ``"s"`` for seconds, ``"ms"`` for milliseconds, ``"us"`` for microseconds.
        """
        if unit == "ms":
            return DatetimeExpr(self._expr, DatetimeOp.EPOCH_MS)
        return DatetimeExpr(self._expr, DatetimeOp.EPOCH)


    def truncate(self, unit: str) -> Expr:
        """
        Truncate datetime to the specified unit.

        Args:
            unit: ``"1h"``, ``"1d"``, ``"1w"``, ``"1mo"``, etc.
        """
        return DatetimeExpr(self._expr, DatetimeOp.TRUNCATE, (unit,))

    def round(self, unit: str) -> Expr:
        """Round datetime to the nearest unit."""
        return DatetimeExpr(self._expr, DatetimeOp.ROUND, (unit,))

    def ceil(self, unit: str) -> Expr:
        """Ceil datetime to the next unit boundary."""
        return DatetimeExpr(self._expr, DatetimeOp.CEIL, (unit,))


    def offset_by(self, duration: str) -> Expr:
        """
        Add a duration offset.

        Args:
            duration: Duration string, e.g., ``"1d"``, ``"2h30m"``, ``"-7d"``.
        """
        return DatetimeExpr(self._expr, DatetimeOp.OFFSET_BY, (duration,))


    def strftime(self, fmt: str) -> Expr:
        """
        Format datetime as a string.

        Args:
            fmt: Format string (e.g., ``"%Y-%m-%d %H:%M:%S"``).
        """
        return DatetimeExpr(self._expr, DatetimeOp.STRFTIME, (fmt,))
