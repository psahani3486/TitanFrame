"""
Projection Pushdown Rule
========================

Pushes projection column selections down to the Scan node.
"""

from titanframe.plan.logical.node import LogicalPlan
from titanframe.plan.logical.projection import Projection
from titanframe.plan.logical.scan import Scan
from titanframe.plan.optimizer.rule import OptimizationRule

class ProjectionPushdown(OptimizationRule):
    @property
    def name(self) -> str:
        return "projection_pushdown"

    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        plan = plan.map_children(self.apply)
        
        if isinstance(plan, Projection):
            child = plan.input
            if isinstance(child, Scan) and child.projection is None:
                cols = []
                for e in plan.exprs:
                    if hasattr(e, "column_name"):
                        cols.append(e.column_name)
                    elif hasattr(e, "name") and hasattr(e, "expr") and hasattr(e.expr, "column_name"):
                        cols.append(e.expr.column_name)
                    else:
                        return plan
                
                new_scan = Scan(
                    source=child.source,
                    format=child.format,
                    schema=child._schema,
                    projection=cols,
                    predicate=child.predicate,
                    limit=child.limit,
                    chunk_size=child.chunk_size
                )
                return Projection(new_scan, plan.exprs)
                
        return plan
