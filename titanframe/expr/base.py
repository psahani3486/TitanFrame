from __future__ import annotations
import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Sequence
from titanframe.core.dtypes import DType
if TYPE_CHECKING:
    pass

class Op(enum.Enum):
    ADD = '+'
    SUB = '-'
    MUL = '*'
    TRUE_DIV = '/'
    FLOOR_DIV = '//'
    MOD = '%'
    POW = '**'
    EQ = '=='
    NE = '!='
    LT = '<'
    LE = '<='
    GT = '>'
    GE = '>='
    AND = '&'
    OR = '|'
    XOR = '^'

class UnaryOp(enum.Enum):
    NEG = '-'
    NOT = '~'
    IS_NULL = 'is_null'
    IS_NOT_NULL = 'is_not_null'
    ABS = 'abs'
    CEIL = 'ceil'
    FLOOR = 'floor'
    SQRT = 'sqrt'
    LOG = 'log'
    EXP = 'exp'

class AggOp(enum.Enum):
    SUM = 'sum'
    MEAN = 'mean'
    MIN = 'min'
    MAX = 'max'
    COUNT = 'count'
    COUNT_DISTINCT = 'count_distinct'
    FIRST = 'first'
    LAST = 'last'
    STD = 'std'
    VAR = 'var'
    MEDIAN = 'median'
    QUANTILE = 'quantile'
    ANY = 'any'
    ALL = 'all'

class SortOrder(enum.Enum):
    ASC = 'asc'
    DESC = 'desc'

class Expr(ABC):
    __slots__ = ()

    @abstractmethod
    def children(self) -> list[Expr]:
        ...

    @abstractmethod
    def display(self, indent: int=0) -> str:
        ...

    def __repr__(self) -> str:
        return self.display()

    def required_columns(self) -> set[str]:
        cols: set[str] = set()
        self._collect_columns(cols)
        return cols

    def _collect_columns(self, out: set[str]) -> None:
        for child in self.children():
            child._collect_columns(out)

    def walk(self) -> list[Expr]:
        result: list[Expr] = [self]
        for child in self.children():
            result.extend(child.walk())
        return result

    def transform(self, fn: Any) -> Expr:
        new_children = [child.transform(fn) for child in self.children()]
        new_self = self._with_children(new_children)
        return fn(new_self)

    @abstractmethod
    def _with_children(self, new_children: list[Expr]) -> Expr:
        ...

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

    def __eq__(self, other: Any) -> Expr:
        return BinaryExpr(Op.EQ, self, _wrap(other))

    def __ne__(self, other: Any) -> Expr:
        return BinaryExpr(Op.NE, self, _wrap(other))

    def __lt__(self, other: Any) -> Expr:
        return BinaryExpr(Op.LT, self, _wrap(other))

    def __le__(self, other: Any) -> Expr:
        return BinaryExpr(Op.LE, self, _wrap(other))

    def __gt__(self, other: Any) -> Expr:
        return BinaryExpr(Op.GT, self, _wrap(other))

    def __ge__(self, other: Any) -> Expr:
        return BinaryExpr(Op.GE, self, _wrap(other))

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

    def __neg__(self) -> Expr:
        return UnaryExpr(UnaryOp.NEG, self)

    def __invert__(self) -> Expr:
        return UnaryExpr(UnaryOp.NOT, self)

    def __abs__(self) -> Expr:
        return UnaryExpr(UnaryOp.ABS, self)

    def alias(self, name: str) -> Expr:
        return AliasExpr(self, name)

    def cast(self, dtype: DType) -> Expr:
        return CastExpr(self, dtype)

    def is_null(self) -> Expr:
        return UnaryExpr(UnaryOp.IS_NULL, self)

    def is_not_null(self) -> Expr:
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

    def sum(self) -> Expr:
        return AggExpr(AggOp.SUM, self)

    def mean(self) -> Expr:
        return AggExpr(AggOp.MEAN, self)

    def min(self) -> Expr:
        return AggExpr(AggOp.MIN, self)

    def max(self) -> Expr:
        return AggExpr(AggOp.MAX, self)

    def count(self) -> Expr:
        return AggExpr(AggOp.COUNT, self)

    def count_distinct(self) -> Expr:
        return AggExpr(AggOp.COUNT_DISTINCT, self)

    def first(self) -> Expr:
        return AggExpr(AggOp.FIRST, self)

    def last(self) -> Expr:
        return AggExpr(AggOp.LAST, self)

    def std(self) -> Expr:
        return AggExpr(AggOp.STD, self)

    def var(self) -> Expr:
        return AggExpr(AggOp.VAR, self)

    def median(self) -> Expr:
        return AggExpr(AggOp.MEDIAN, self)

    @property
    def str(self) -> Any:
        from titanframe.expr.string_expr import StringAccessor
        return StringAccessor(self)

    @property
    def dt(self) -> Any:
        from titanframe.expr.datetime_expr import DatetimeAccessor
        return DatetimeAccessor(self)

    @property
    def window(self) -> Any:
        from titanframe.expr.window_expr import WindowAccessor
        return WindowAccessor(self)

    def over(self, *partition_by: Any) -> Any:
        from titanframe.expr.window_expr import WindowExpr, DEFAULT_FRAME
        from titanframe.expr.column_expr import col as _col
        parts = [_col(p) if isinstance(p, str) else p for p in partition_by]
        if isinstance(self, AggExpr):
            return WindowExpr(self.child, partition_by=parts, agg_op=self.op)
        return WindowExpr(self, partition_by=parts)

    def map(self, func: Any, return_dtype: Optional[DType]=None) -> Expr:
        from titanframe.expr.udf_expr import UDFExpr, UDFType
        return UDFExpr(self, func, UDFType.SCALAR, return_dtype)

    def map_batches(self, func: Any, return_dtype: Optional[DType]=None) -> Expr:
        from titanframe.expr.udf_expr import UDFExpr, UDFType
        return UDFExpr(self, func, UDFType.VECTORIZED, return_dtype)

    def asc(self) -> SortExpr:
        return SortExpr(self, SortOrder.ASC)

    def desc(self) -> SortExpr:
        return SortExpr(self, SortOrder.DESC)

    @abstractmethod
    def __hash__(self) -> int:
        ...

class BinaryExpr(Expr):
    __slots__ = ('op', 'left', 'right')

    def __init__(self, op: Op, left: Expr, right: Expr):
        self.op = op
        self.left = left
        self.right = right

    def children(self) -> list[Expr]:
        return [self.left, self.right]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return BinaryExpr(self.op, new_children[0], new_children[1])

    def display(self, indent: int=0) -> str:
        return f'BinaryExpr({self.op.name}, {self.left.display()}, {self.right.display()})'

    def __hash__(self) -> int:
        return hash(('binary', self.op, self.left, self.right))

class UnaryExpr(Expr):
    __slots__ = ('op', 'operand')

    def __init__(self, op: UnaryOp, operand: Expr):
        self.op = op
        self.operand = operand

    def children(self) -> list[Expr]:
        return [self.operand]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return UnaryExpr(self.op, new_children[0])

    def display(self, indent: int=0) -> str:
        return f'UnaryExpr({self.op.name}, {self.operand.display()})'

    def __hash__(self) -> int:
        return hash(('unary', self.op, self.operand))

class AggExpr(Expr):
    __slots__ = ('op', 'child')

    def __init__(self, op: AggOp, child: Expr):
        self.op = op
        self.child = child

    def children(self) -> list[Expr]:
        return [self.child]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return AggExpr(self.op, new_children[0])

    def display(self, indent: int=0) -> str:
        return f'AggExpr({self.op.name}, {self.child.display()})'

    def __hash__(self) -> int:
        return hash(('agg', self.op, self.child))

class CastExpr(Expr):
    __slots__ = ('child', 'target_dtype')

    def __init__(self, child: Expr, target_dtype: DType):
        self.child = child
        self.target_dtype = target_dtype

    def children(self) -> list[Expr]:
        return [self.child]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return CastExpr(new_children[0], self.target_dtype)

    def display(self, indent: int=0) -> str:
        return f'CastExpr({self.child.display()}, {self.target_dtype})'

    def __hash__(self) -> int:
        return hash(('cast', self.child, self.target_dtype))

class AliasExpr(Expr):
    __slots__ = ('child', 'name')

    def __init__(self, child: Expr, name: str):
        self.child = child
        self.name = name

    def children(self) -> list[Expr]:
        return [self.child]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return AliasExpr(new_children[0], self.name)

    def display(self, indent: int=0) -> str:
        return f'AliasExpr({self.child.display()}, {self.name!r})'

    def __hash__(self) -> int:
        return hash(('alias', self.child, self.name))

class SortExpr(Expr):
    __slots__ = ('child', 'order')

    def __init__(self, child: Expr, order: SortOrder):
        self.child = child
        self.order = order

    def children(self) -> list[Expr]:
        return [self.child]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return SortExpr(new_children[0], self.order)

    def display(self, indent: int=0) -> str:
        return f'SortExpr({self.child.display()}, {self.order.name})'

    def __hash__(self) -> int:
        return hash(('sort', self.child, self.order))

def _wrap(value: Any) -> Expr:
    if isinstance(value, Expr):
        return value
    from titanframe.expr.literal_expr import lit
    return lit(value)
