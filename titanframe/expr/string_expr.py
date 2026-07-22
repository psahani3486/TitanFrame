from __future__ import annotations
import enum
from typing import Optional
from titanframe.expr.base import Expr

class StringOp(enum.Enum):
    LOWER = 'lower'
    UPPER = 'upper'
    STRIP = 'strip'
    LSTRIP = 'lstrip'
    RSTRIP = 'rstrip'
    LENGTH = 'length'
    CONTAINS = 'contains'
    STARTS_WITH = 'starts_with'
    ENDS_WITH = 'ends_with'
    REPLACE = 'replace'
    SLICE = 'slice'
    SPLIT = 'split'
    CONCAT = 'concat'
    PAD = 'pad'
    REPEAT = 'repeat'
    REVERSE = 'reverse'
    TO_LOWERCASE = 'to_lowercase'
    TO_UPPERCASE = 'to_uppercase'
    TRIM = 'trim'
    IS_ALPHANUMERIC = 'is_alphanumeric'
    IS_NUMERIC = 'is_numeric'
    IS_ALPHA = 'is_alpha'
    COUNT_MATCHES = 'count_matches'
    EXTRACT = 'extract'
    JSON_PATH = 'json_path'

class StringExpr(Expr):
    __slots__ = ('child', 'op', 'args')

    def __init__(self, child: Expr, op: StringOp, args: tuple=()):
        self.child = child
        self.op = op
        self.args = args

    def children(self) -> list[Expr]:
        return [self.child]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return StringExpr(new_children[0], self.op, self.args)

    def display(self, indent: int=0) -> str:
        args_str = f', {self.args}' if self.args else ''
        return f'StringExpr({self.op.value}, {self.child.display()}{args_str})'

    def __hash__(self) -> int:
        return hash(('string', self.child, self.op, self.args))

class StringAccessor:
    __slots__ = ('_expr',)

    def __init__(self, expr: Expr):
        self._expr = expr

    def lower(self) -> Expr:
        return StringExpr(self._expr, StringOp.LOWER)

    def upper(self) -> Expr:
        return StringExpr(self._expr, StringOp.UPPER)

    def strip(self, chars: Optional[str]=None) -> Expr:
        args = (chars,) if chars else ()
        return StringExpr(self._expr, StringOp.STRIP, args)

    def lstrip(self, chars: Optional[str]=None) -> Expr:
        args = (chars,) if chars else ()
        return StringExpr(self._expr, StringOp.LSTRIP, args)

    def rstrip(self, chars: Optional[str]=None) -> Expr:
        args = (chars,) if chars else ()
        return StringExpr(self._expr, StringOp.RSTRIP, args)

    def length(self) -> Expr:
        return StringExpr(self._expr, StringOp.LENGTH)

    def contains(self, pattern: str, regex: bool=False) -> Expr:
        return StringExpr(self._expr, StringOp.CONTAINS, (pattern, regex))

    def starts_with(self, prefix: str) -> Expr:
        return StringExpr(self._expr, StringOp.STARTS_WITH, (prefix,))

    def ends_with(self, suffix: str) -> Expr:
        return StringExpr(self._expr, StringOp.ENDS_WITH, (suffix,))

    def replace(self, pattern: str, replacement: str, regex: bool=False) -> Expr:
        return StringExpr(self._expr, StringOp.REPLACE, (pattern, replacement, regex))

    def slice(self, start: int, length: Optional[int]=None) -> Expr:
        return StringExpr(self._expr, StringOp.SLICE, (start, length))

    def split(self, separator: str) -> Expr:
        return StringExpr(self._expr, StringOp.SPLIT, (separator,))

    def repeat(self, n: int) -> Expr:
        return StringExpr(self._expr, StringOp.REPEAT, (n,))

    def reverse(self) -> Expr:
        return StringExpr(self._expr, StringOp.REVERSE)

    def pad(self, width: int, side: str='left', fillchar: str=' ') -> Expr:
        return StringExpr(self._expr, StringOp.PAD, (width, side, fillchar))

    def is_alphanumeric(self) -> Expr:
        return StringExpr(self._expr, StringOp.IS_ALPHANUMERIC)

    def is_numeric(self) -> Expr:
        return StringExpr(self._expr, StringOp.IS_NUMERIC)

    def is_alpha(self) -> Expr:
        return StringExpr(self._expr, StringOp.IS_ALPHA)

    def count_matches(self, pattern: str) -> Expr:
        return StringExpr(self._expr, StringOp.COUNT_MATCHES, (pattern,))

    def extract(self, pattern: str, group_index: int=0) -> Expr:
        return StringExpr(self._expr, StringOp.EXTRACT, (pattern, group_index))

    def json_path(self, path: str) -> Expr:
        return StringExpr(self._expr, StringOp.JSON_PATH, (path,))
