from __future__ import annotations
import enum
from typing import Optional
from titanframe.expr.base import Expr

class DatetimeOp(enum.Enum):
    YEAR = 'year'
    MONTH = 'month'
    DAY = 'day'
    HOUR = 'hour'
    MINUTE = 'minute'
    SECOND = 'second'
    MILLISECOND = 'millisecond'
    MICROSECOND = 'microsecond'
    NANOSECOND = 'nanosecond'
    WEEKDAY = 'weekday'
    ISO_WEEKDAY = 'iso_weekday'
    WEEK = 'week'
    DAY_OF_YEAR = 'day_of_year'
    QUARTER = 'quarter'
    DATE = 'date'
    TIME = 'time'
    EPOCH = 'epoch'
    EPOCH_MS = 'epoch_ms'
    TRUNCATE = 'truncate'
    ROUND = 'round'
    CEIL = 'ceil'
    OFFSET_BY = 'offset_by'
    DIFF = 'diff'
    STRFTIME = 'strftime'
    TIMESTAMP = 'timestamp'

class DatetimeExpr(Expr):
    __slots__ = ('child', 'op', 'args')

    def __init__(self, child: Expr, op: DatetimeOp, args: tuple=()):
        self.child = child
        self.op = op
        self.args = args

    def children(self) -> list[Expr]:
        return [self.child]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return DatetimeExpr(new_children[0], self.op, self.args)

    def display(self, indent: int=0) -> str:
        args_str = f', {self.args}' if self.args else ''
        return f'DatetimeExpr({self.op.value}, {self.child.display()}{args_str})'

    def __hash__(self) -> int:
        return hash(('datetime', self.child, self.op, self.args))

class DatetimeAccessor:
    __slots__ = ('_expr',)

    def __init__(self, expr: Expr):
        self._expr = expr

    @property
    def year(self) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.YEAR)

    @property
    def month(self) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.MONTH)

    @property
    def day(self) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.DAY)

    @property
    def hour(self) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.HOUR)

    @property
    def minute(self) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.MINUTE)

    @property
    def second(self) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.SECOND)

    @property
    def millisecond(self) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.MILLISECOND)

    @property
    def microsecond(self) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.MICROSECOND)

    @property
    def nanosecond(self) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.NANOSECOND)

    @property
    def weekday(self) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.WEEKDAY)

    @property
    def iso_weekday(self) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.ISO_WEEKDAY)

    @property
    def week(self) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.WEEK)

    @property
    def day_of_year(self) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.DAY_OF_YEAR)

    @property
    def quarter(self) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.QUARTER)

    @property
    def date(self) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.DATE)

    @property
    def time(self) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.TIME)

    def epoch(self, unit: str='s') -> Expr:
        if unit == 'ms':
            return DatetimeExpr(self._expr, DatetimeOp.EPOCH_MS)
        return DatetimeExpr(self._expr, DatetimeOp.EPOCH)

    def truncate(self, unit: str) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.TRUNCATE, (unit,))

    def round(self, unit: str) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.ROUND, (unit,))

    def ceil(self, unit: str) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.CEIL, (unit,))

    def offset_by(self, duration: str) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.OFFSET_BY, (duration,))

    def strftime(self, fmt: str) -> Expr:
        return DatetimeExpr(self._expr, DatetimeOp.STRFTIME, (fmt,))
