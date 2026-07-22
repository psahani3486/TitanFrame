"""
Slice Pushdown Rule
===================

Pushes LIMIT / OFFSET down to the SCAN node to avoid reading unnecessary rows.
"""

from titanframe.plan.logical.node import LogicalPlan
from titanframe.plan.logical.limit import Limit
from titanframe.plan.logical.scan import Scan
from titanframe.plan.logical.projection import Projection
from titanframe.plan.optimizer.rule import OptimizationRule


class SlicePushdown(OptimizationRule):
    @property
    def name(self) -> str:
        return "slice_pushdown"

    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        plan = plan.map_children(self.apply)
        
        if isinstance(plan, Limit):
            child = plan.input
            
            if isinstance(child, Projection):
                new_limit = Limit(child.input, plan.limit, plan.offset)
                pushed = self.apply(new_limit)
                return Projection(pushed, child.exprs)
                
            if isinstance(child, Scan):
                scan_limit = plan.limit
                if plan.limit is not None and plan.offset > 0:
                    scan_limit = plan.limit + plan.offset
                elif plan.limit is None and plan.offset > 0:
                    scan_limit = None
                    
                if scan_limit is not None:
                    new_limit = scan_limit
                    if child.limit is not None:
                        new_limit = min(scan_limit, child.limit)
                        
                    new_scan = Scan(
                        source=child.source,
                        format=child.format,
                        schema=child._schema,
                        projection=child.projection,
                        predicate=child.predicate,
                        limit=new_limit,
                        chunk_size=child.chunk_size
                    )
                    
                    if plan.offset == 0 and plan.limit is not None:
                        return new_scan
                    else:
                        return Limit(new_scan, plan.limit, plan.offset)
                        
        return plan
