import pytest

from titanframe.core.dtypes import Int64, Utf8, Bool
from titanframe.core.schema import Schema
from titanframe.expr.column_expr import col
from titanframe.expr.literal_expr import lit
from titanframe.plan.logical.scan import Scan, ScanFormat
from titanframe.plan.logical.projection import Projection
from titanframe.plan.logical.filter import Filter
from titanframe.plan.logical.limit import Limit
from titanframe.plan.optimizer.driver import QueryOptimizer
from titanframe.plan.optimizer.constant_folding import ConstantFolding
from titanframe.plan.optimizer.slice_pushdown import SlicePushdown
from titanframe.plan.optimizer.predicate_pushdown import PredicatePushdown
from titanframe.plan.optimizer.projection_pushdown import ProjectionPushdown
from titanframe.plan.optimizer.fusion import OperatorFusion


@pytest.fixture
def scan_node():
    schema = Schema({"id": Int64, "name": Utf8, "age": Int64})
    return Scan("users.parquet", ScanFormat.PARQUET, schema)


def test_constant_folding(scan_node):
    expr = lit(2) + lit(3)
    plan = Filter(scan_node, expr)
    
    optimizer = QueryOptimizer([ConstantFolding()])
    opt_plan = optimizer.optimize(plan)
    
    assert isinstance(opt_plan, Filter)
    assert opt_plan.predicate.value == 5


def test_slice_pushdown(scan_node):
    plan = Limit(scan_node, limit=10, offset=0)
    
    optimizer = QueryOptimizer([SlicePushdown()])
    opt_plan = optimizer.optimize(plan)
    
    assert isinstance(opt_plan, Scan)
    assert opt_plan.limit == 10


def test_slice_pushdown_with_offset(scan_node):
    plan = Limit(scan_node, limit=10, offset=5)
    
    optimizer = QueryOptimizer([SlicePushdown()])
    opt_plan = optimizer.optimize(plan)
    
    assert isinstance(opt_plan, Limit)
    assert opt_plan.limit == 10
    assert opt_plan.offset == 5
    assert isinstance(opt_plan.input, Scan)
    assert opt_plan.input.limit == 15


def test_predicate_pushdown(scan_node):
    expr = col("age") > lit(18)
    plan = Filter(scan_node, expr)
    
    optimizer = QueryOptimizer([PredicatePushdown()])
    opt_plan = optimizer.optimize(plan)
    
    assert isinstance(opt_plan, Scan)
    assert opt_plan.predicate is not None


def test_projection_pushdown(scan_node):
    plan = Projection(scan_node, [col("id")])
    
    optimizer = QueryOptimizer([ProjectionPushdown()])
    opt_plan = optimizer.optimize(plan)
    
    assert isinstance(opt_plan, Projection)
    assert isinstance(opt_plan.input, Scan)
    assert opt_plan.input.projection == ["id"]


def test_operator_fusion(scan_node):
    expr1 = col("age") > lit(18)
    expr2 = col("name") == lit("Alice")
    plan = Filter(Filter(scan_node, expr1), expr2)
    
    optimizer = QueryOptimizer([OperatorFusion()])
    opt_plan = optimizer.optimize(plan)
    
    assert isinstance(opt_plan, Filter)
    assert isinstance(opt_plan.input, Scan)
