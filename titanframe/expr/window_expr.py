from __future__ import annotations
import enum
from typing import Optional, Sequence
from titanframe.expr.base import Expr, AggOp

class WindowFrameType(enum.Enum):
    ROWS = 'rows'
    RANGE = 'range'
    GROUPS = 'groups'

class WindowFrame:
    __slots__ = ('frame_type', 'start', 'end')

    def __init__(self, frame_type: WindowFrameType=WindowFrameType.ROWS, start: Optional[int]=None, end: Optional[int]=0):
        self.frame_type = frame_type
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        start_str = 'UNBOUNDED PRECEDING' if self.start is None else 'CURRENT ROW' if self.start == 0 else f'{abs(self.start)} PRECEDING' if self.start < 0 else f'{self.start} FOLLOWING'
        end_str = 'UNBOUNDED FOLLOWING' if self.end is None else 'CURRENT ROW' if self.end == 0 else f'{abs(self.end)} PRECEDING' if self.end < 0 else f'{self.end} FOLLOWING'
        return f'{self.frame_type.value}({start_str} TO {end_str})'

    def __hash__(self) -> int:
        return hash((self.frame_type, self.start, self.end))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WindowFrame):
            return NotImplemented
        return (self.frame_type, self.start, self.end) == (other.frame_type, other.start, other.end)
DEFAULT_FRAME = WindowFrame(WindowFrameType.ROWS, start=None, end=0)
FULL_FRAME = WindowFrame(WindowFrameType.ROWS, start=None, end=None)

class WindowOp(enum.Enum):
    ROW_NUMBER = 'row_number'
    RANK = 'rank'
    DENSE_RANK = 'dense_rank'
    PERCENT_RANK = 'percent_rank'
    CUME_DIST = 'cume_dist'
    NTILE = 'ntile'
    LAG = 'lag'
    LEAD = 'lead'
    FIRST_VALUE = 'first_value'
    LAST_VALUE = 'last_value'
    NTH_VALUE = 'nth_value'

class WindowExpr(Expr):
    __slots__ = ('child', 'partition_by', 'order_by', 'frame', 'window_op', 'agg_op', 'args')

    def __init__(self, child: Optional[Expr], partition_by: Sequence[Expr]=(), order_by: Sequence[Expr]=(), frame: WindowFrame=DEFAULT_FRAME, window_op: Optional[WindowOp]=None, agg_op: Optional[AggOp]=None, args: tuple=()):
        self.child = child
        self.partition_by = list(partition_by)
        self.order_by = list(order_by)
        self.frame = frame
        self.window_op = window_op
        self.agg_op = agg_op
        self.args = args

    def children(self) -> list[Expr]:
        result: list[Expr] = []
        if self.child is not None:
            result.append(self.child)
        result.extend(self.partition_by)
        result.extend(self.order_by)
        return result

    def _with_children(self, new_children: list[Expr]) -> Expr:
        idx = 0
        new_child = None
        if self.child is not None:
            new_child = new_children[idx]
            idx += 1
        n_part = len(self.partition_by)
        new_partition = new_children[idx:idx + n_part]
        idx += n_part
        new_order = new_children[idx:]
        return WindowExpr(new_child, new_partition, new_order, self.frame, self.window_op, self.agg_op, self.args)

    def display(self, indent: int=0) -> str:
        op_name = self.window_op.value if self.window_op else self.agg_op.value if self.agg_op else 'unknown'
        child_str = self.child.display() if self.child else ''
        parts = [f'op={op_name}']
        if child_str:
            parts.append(f'child={child_str}')
        if self.partition_by:
            parts.append(f"partition_by=[{', '.join((e.display() for e in self.partition_by))}]")
        if self.order_by:
            parts.append(f"order_by=[{', '.join((e.display() for e in self.order_by))}]")
        parts.append(f'frame={self.frame}')
        return f"WindowExpr({', '.join(parts)})"

    def __hash__(self) -> int:
        return hash(('window', self.child, tuple(self.partition_by), tuple(self.order_by), self.frame, self.window_op, self.agg_op, self.args))

class WindowAccessor:
    __slots__ = ('_expr',)

    def __init__(self, expr: Expr):
        self._expr = expr

    def row_number(self) -> _WindowBuilder:
        return _WindowBuilder(None, window_op=WindowOp.ROW_NUMBER)

    def rank(self) -> _WindowBuilder:
        return _WindowBuilder(None, window_op=WindowOp.RANK)

    def dense_rank(self) -> _WindowBuilder:
        return _WindowBuilder(None, window_op=WindowOp.DENSE_RANK)

    def lag(self, offset: int=1, default: object=None) -> _WindowBuilder:
        return _WindowBuilder(self._expr, window_op=WindowOp.LAG, args=(offset, default))

    def lead(self, offset: int=1, default: object=None) -> _WindowBuilder:
        return _WindowBuilder(self._expr, window_op=WindowOp.LEAD, args=(offset, default))

    def first_value(self) -> _WindowBuilder:
        return _WindowBuilder(self._expr, window_op=WindowOp.FIRST_VALUE)

    def last_value(self) -> _WindowBuilder:
        return _WindowBuilder(self._expr, window_op=WindowOp.LAST_VALUE)

    def ntile(self, n: int) -> _WindowBuilder:
        return _WindowBuilder(None, window_op=WindowOp.NTILE, args=(n,))

class _WindowBuilder:
    __slots__ = ('_child', '_window_op', '_agg_op', '_args', '_partition_by', '_order_by', '_frame')

    def __init__(self, child: Optional[Expr], window_op: Optional[WindowOp]=None, agg_op: Optional[AggOp]=None, args: tuple=()):
        self._child = child
        self._window_op = window_op
        self._agg_op = agg_op
        self._args = args
        self._partition_by: list[Expr] = []
        self._order_by: list[Expr] = []
        self._frame = DEFAULT_FRAME

    def over(self, *partition_by: Expr) -> _WindowBuilder:
        from titanframe.expr.column_expr import col, ColumnExpr
        self._partition_by = [col(p) if isinstance(p, str) else p for p in partition_by]
        return self

    def order_by(self, *exprs: Expr) -> _WindowBuilder:
        from titanframe.expr.column_expr import col
        self._order_by = [col(e) if isinstance(e, str) else e for e in exprs]
        return self

    def frame(self, frame: WindowFrame) -> _WindowBuilder:
        self._frame = frame
        return self

    def build(self) -> WindowExpr:
        return WindowExpr(self._child, self._partition_by, self._order_by, self._frame, self._window_op, self._agg_op, self._args)

    def __repr__(self) -> str:
        return repr(self.build())
