"""
Expression Evaluator
====================

Evaluates logical expressions into PyArrow arrays over a Chunk.
"""
import pyarrow as pa
import pyarrow.compute as pc

from titanframe.plan.physical.node import Chunk
from titanframe.expr.base import Expr, BinaryExpr, UnaryExpr, CastExpr, Op
from titanframe.expr.column_expr import ColumnExpr
from titanframe.expr.literal_expr import LiteralExpr

class ExprEvaluator:
    """Walks an Expr tree and computes results on a Chunk."""
    
    def eval(self, expr: Expr, chunk: Chunk) -> pa.Array | pa.Scalar:
        if hasattr(expr, "expr") and isinstance(expr.expr, Expr):
            expr = expr.expr
            
        if isinstance(expr, ColumnExpr):
            return chunk.column(expr.column_name)
            
        elif isinstance(expr, LiteralExpr):
            return pa.scalar(expr.value, type=expr.dtype.arrow_type)
            
        elif isinstance(expr, BinaryExpr):
            l = self.eval(expr.left, chunk)
            r = self.eval(expr.right, chunk)
            return self._apply_binary(expr.op, l, r)
            
        elif isinstance(expr, UnaryExpr):
            child = self.eval(expr.child, chunk)
            return self._apply_unary(expr.op, child)
            
        elif isinstance(expr, CastExpr):
            child = self.eval(expr.child, chunk)
            arrow_type = expr.target_dtype.arrow_type
            return pc.cast(child, target_type=arrow_type, safe=not expr.unsafe)
            
        else:
            raise NotImplementedError(f"Evaluation of {type(expr).__name__} not implemented")

    def _apply_binary(self, op: Op, l, r):
        if op == Op.ADD: return pc.add(l, r)
        if op == Op.SUB: return pc.subtract(l, r)
        if op == Op.MUL: return pc.multiply(l, r)
        if op == Op.TRUE_DIV: return pc.divide(l, r)
        if op == Op.FLOOR_DIV: return pc.divide(l, r)
        if op == Op.EQ: return pc.equal(l, r)
        if op == Op.NE: return pc.not_equal(l, r)
        if op == Op.GT: return pc.greater(l, r)
        if op == Op.GE: return pc.greater_equal(l, r)
        if op == Op.LT: return pc.less(l, r)
        if op == Op.LE: return pc.less_equal(l, r)
        if op == Op.AND: return pc.and_(l, r)
        if op == Op.OR: return pc.or_(l, r)
        raise NotImplementedError(f"Binary op {op} not implemented in PyArrow")
        
    def _apply_unary(self, op: Op, child):
        if op == Op.NEG: return pc.negate(child)
        if op == Op.INV: return pc.invert(child)
        if op == Op.IS_NULL: return pc.is_null(child)
        if op == Op.IS_NOT_NULL: return pc.is_valid(child)
        raise NotImplementedError(f"Unary op {op} not implemented in PyArrow")
