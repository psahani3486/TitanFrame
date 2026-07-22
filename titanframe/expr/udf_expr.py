from __future__ import annotations
import enum
from typing import Any, Callable, Optional
from titanframe.core.dtypes import DType
from titanframe.expr.base import Expr

class UDFType(enum.Enum):
    SCALAR = 'scalar'
    VECTORIZED = 'vectorized'

class UDFExpr(Expr):
    __slots__ = ('child', 'func', 'udf_type', 'return_dtype', 'func_name')

    def __init__(self, child: Expr, func: Callable[..., Any], udf_type: UDFType=UDFType.SCALAR, return_dtype: Optional[DType]=None, func_name: Optional[str]=None):
        self.child = child
        self.func = func
        self.udf_type = udf_type
        self.return_dtype = return_dtype
        self.func_name = func_name or getattr(func, '__name__', '<lambda>')

    def children(self) -> list[Expr]:
        return [self.child]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return UDFExpr(new_children[0], self.func, self.udf_type, self.return_dtype, self.func_name)

    def display(self, indent: int=0) -> str:
        return f'UDFExpr({self.udf_type.value}, func={self.func_name!r}, child={self.child.display()}, return_dtype={self.return_dtype})'

    def __hash__(self) -> int:
        return hash(('udf', id(self.func), self.child, self.udf_type))

class MultiColumnUDFExpr(Expr):
    __slots__ = ('inputs', 'func', 'udf_type', 'return_dtype', 'func_name')

    def __init__(self, inputs: list[Expr], func: Callable[..., Any], udf_type: UDFType=UDFType.SCALAR, return_dtype: Optional[DType]=None, func_name: Optional[str]=None):
        self.inputs = list(inputs)
        self.func = func
        self.udf_type = udf_type
        self.return_dtype = return_dtype
        self.func_name = func_name or getattr(func, '__name__', '<lambda>')

    def children(self) -> list[Expr]:
        return list(self.inputs)

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return MultiColumnUDFExpr(new_children, self.func, self.udf_type, self.return_dtype, self.func_name)

    def display(self, indent: int=0) -> str:
        inputs_str = ', '.join((e.display() for e in self.inputs))
        return f'MultiColumnUDFExpr({self.udf_type.value}, func={self.func_name!r}, inputs=[{inputs_str}], return_dtype={self.return_dtype})'

    def __hash__(self) -> int:
        return hash(('multi_udf', id(self.func), tuple(self.inputs), self.udf_type))
