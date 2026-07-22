"""
String Expression Accessor — ``col("name").str.contains(...)``
================================================================

Provides string-specific operations as a namespace accessor on expressions,
mirroring Pandas' ``Series.str`` accessor.

All methods return new Expr nodes that are evaluated lazily during execution.

Example::

    >>> from titanframe import col
    >>> expr = col("city").str.lower()
    >>> expr = col("email").str.contains("@gmail")
"""

from __future__ import annotations

import enum
from typing import Optional

from titanframe.expr.base import Expr


class StringOp(enum.Enum):
    """String operations."""
    LOWER = "lower"
    UPPER = "upper"
    STRIP = "strip"
    LSTRIP = "lstrip"
    RSTRIP = "rstrip"
    LENGTH = "length"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REPLACE = "replace"
    SLICE = "slice"
    SPLIT = "split"
    CONCAT = "concat"
    PAD = "pad"
    REPEAT = "repeat"
    REVERSE = "reverse"
    TO_LOWERCASE = "to_lowercase"
    TO_UPPERCASE = "to_uppercase"
    TRIM = "trim"
    IS_ALPHANUMERIC = "is_alphanumeric"
    IS_NUMERIC = "is_numeric"
    IS_ALPHA = "is_alpha"
    COUNT_MATCHES = "count_matches"
    EXTRACT = "extract"
    JSON_PATH = "json_path"


class StringExpr(Expr):
    """
    A string operation applied to an expression.

    Attributes:
        child: The input expression (must produce string values).
        op: The string operation to apply.
        args: Additional arguments (pattern, replacement, etc.).
    """

    __slots__ = ("child", "op", "args")

    def __init__(self, child: Expr, op: StringOp, args: tuple = ()):
        self.child = child
        self.op = op
        self.args = args

    def children(self) -> list[Expr]:
        return [self.child]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return StringExpr(new_children[0], self.op, self.args)

    def display(self, indent: int = 0) -> str:
        args_str = f", {self.args}" if self.args else ""
        return f"StringExpr({self.op.value}, {self.child.display()}{args_str})"

    def __hash__(self) -> int:
        return hash(("string", self.child, self.op, self.args))


class StringAccessor:
    """
    Namespace accessor for string operations on an expression.

    Accessed via ``expr.str`` — mirrors the Pandas ``Series.str`` API.

    Example::

        >>> col("name").str.lower()
        >>> col("email").str.contains("@")
        >>> col("text").str.replace("old", "new")
    """

    __slots__ = ("_expr",)

    def __init__(self, expr: Expr):
        self._expr = expr

    def lower(self) -> Expr:
        """Convert to lowercase."""
        return StringExpr(self._expr, StringOp.LOWER)

    def upper(self) -> Expr:
        """Convert to uppercase."""
        return StringExpr(self._expr, StringOp.UPPER)

    def strip(self, chars: Optional[str] = None) -> Expr:
        """Strip whitespace (or specified chars) from both ends."""
        args = (chars,) if chars else ()
        return StringExpr(self._expr, StringOp.STRIP, args)

    def lstrip(self, chars: Optional[str] = None) -> Expr:
        """Strip from left end."""
        args = (chars,) if chars else ()
        return StringExpr(self._expr, StringOp.LSTRIP, args)

    def rstrip(self, chars: Optional[str] = None) -> Expr:
        """Strip from right end."""
        args = (chars,) if chars else ()
        return StringExpr(self._expr, StringOp.RSTRIP, args)

    def length(self) -> Expr:
        """UTF-8 character length."""
        return StringExpr(self._expr, StringOp.LENGTH)

    def contains(self, pattern: str, regex: bool = False) -> Expr:
        """Check if string contains a pattern."""
        return StringExpr(self._expr, StringOp.CONTAINS, (pattern, regex))

    def starts_with(self, prefix: str) -> Expr:
        """Check if string starts with prefix."""
        return StringExpr(self._expr, StringOp.STARTS_WITH, (prefix,))

    def ends_with(self, suffix: str) -> Expr:
        """Check if string ends with suffix."""
        return StringExpr(self._expr, StringOp.ENDS_WITH, (suffix,))

    def replace(self, pattern: str, replacement: str, regex: bool = False) -> Expr:
        """Replace occurrences of pattern with replacement."""
        return StringExpr(self._expr, StringOp.REPLACE, (pattern, replacement, regex))

    def slice(self, start: int, length: Optional[int] = None) -> Expr:
        """Extract a substring."""
        return StringExpr(self._expr, StringOp.SLICE, (start, length))

    def split(self, separator: str) -> Expr:
        """Split by separator (produces a list column)."""
        return StringExpr(self._expr, StringOp.SPLIT, (separator,))

    def repeat(self, n: int) -> Expr:
        """Repeat the string n times."""
        return StringExpr(self._expr, StringOp.REPEAT, (n,))

    def reverse(self) -> Expr:
        """Reverse the string."""
        return StringExpr(self._expr, StringOp.REVERSE)

    def pad(self, width: int, side: str = "left", fillchar: str = " ") -> Expr:
        """Pad the string to a minimum width."""
        return StringExpr(self._expr, StringOp.PAD, (width, side, fillchar))

    def is_alphanumeric(self) -> Expr:
        """Check if all characters are alphanumeric."""
        return StringExpr(self._expr, StringOp.IS_ALPHANUMERIC)

    def is_numeric(self) -> Expr:
        """Check if all characters are numeric."""
        return StringExpr(self._expr, StringOp.IS_NUMERIC)

    def is_alpha(self) -> Expr:
        """Check if all characters are alphabetic."""
        return StringExpr(self._expr, StringOp.IS_ALPHA)

    def count_matches(self, pattern: str) -> Expr:
        """Count occurrences of a pattern."""
        return StringExpr(self._expr, StringOp.COUNT_MATCHES, (pattern,))

    def extract(self, pattern: str, group_index: int = 0) -> Expr:
        """Extract first regex match."""
        return StringExpr(self._expr, StringOp.EXTRACT, (pattern, group_index))

    def json_path(self, path: str) -> Expr:
        """Extract value from JSON string using JSONPath."""
        return StringExpr(self._expr, StringOp.JSON_PATH, (path,))
