"""
Physical Planner
================

Translates a LogicalPlan to a PhysicalPlan.
"""
from titanframe.plan.logical.node import LogicalPlan
from titanframe.plan.logical.scan import Scan
from titanframe.plan.logical.filter import Filter
from titanframe.plan.logical.projection import Projection
from titanframe.plan.logical.aggregation import Aggregation
from titanframe.plan.logical.join import Join
from titanframe.plan.logical.sort import Sort
from titanframe.plan.logical.limit import Limit
from titanframe.plan.logical.sink import Sink

from titanframe.plan.physical.node import PhysicalPlan
from titanframe.plan.physical.scan_exec import ScanExec
from titanframe.plan.physical.filter_exec import FilterExec
from titanframe.plan.physical.project_exec import ProjectExec
from titanframe.plan.physical.hash_agg_exec import HashAggExec
from titanframe.plan.physical.sort_merge_exec import SortMergeExec
from titanframe.plan.physical.hash_join_exec import HashJoinExec
from titanframe.plan.physical.sink_exec import SinkExec
from titanframe.plan.physical.limit_exec import LimitExec
from titanframe.plan.physical.union_exec import UnionExec

class PhysicalPlanner:
    def plan(self, logical_plan: LogicalPlan) -> PhysicalPlan:
        if isinstance(logical_plan, Scan):
            return ScanExec(
                source=logical_plan.source,
                format=logical_plan.format,
                columns=logical_plan.projection,
                limit=logical_plan.limit,
                table=getattr(logical_plan, 'table', None)
            )
            
        if isinstance(logical_plan, Filter):
            return FilterExec(
                input=self.plan(logical_plan.input),
                predicate=logical_plan.predicate
            )
            
        if isinstance(logical_plan, Projection):
            return ProjectExec(
                input=self.plan(logical_plan.input),
                exprs=logical_plan.exprs,
                output_names=logical_plan.output_schema().names
            )
            
        if isinstance(logical_plan, Aggregation):
            return HashAggExec(
                input=self.plan(logical_plan.input),
                group_by=logical_plan.group_keys,
                aggs=logical_plan.agg_exprs,
                output_names=logical_plan.output_schema().names
            )
            
        if isinstance(logical_plan, Sort):
            return SortMergeExec(
                input=self.plan(logical_plan.input),
                sort_keys=logical_plan.sort_exprs
            )
            
        if isinstance(logical_plan, Join):
            keys = []
            for k in logical_plan.on:
                if isinstance(k, str): keys.append(k)
                elif hasattr(k, "column_name"): keys.append(k.column_name)
                elif hasattr(k, "expr"): keys.append(k.expr.column_name)
                else: keys.append(str(k))
                
            return HashJoinExec(
                left=self.plan(logical_plan.left),
                right=self.plan(logical_plan.right),
                keys=keys,
                how=logical_plan.how,
                suffix=logical_plan.suffix
            )
            
        if isinstance(logical_plan, Limit):
            return LimitExec(
                input=self.plan(logical_plan.input),
                limit=logical_plan.limit,
                offset=logical_plan.offset
            )
            
        if isinstance(logical_plan, Sink):
            return SinkExec(
                input=self.plan(logical_plan.input),
                target=logical_plan.target,
                format=logical_plan.format
            )
            
        from titanframe.plan.logical.union import Union
        if isinstance(logical_plan, Union):
            return UnionExec(
                inputs=[self.plan(child) for child in logical_plan.inputs]
            )
            
        from titanframe.plan.logical.distinct import Distinct
        if isinstance(logical_plan, Distinct):
            return HashAggExec(
                input=self.plan(logical_plan.input),
                group_by=logical_plan.subset if logical_plan.subset else logical_plan.input.output_schema().names,
                aggs=[],
                output_names=logical_plan.output_schema().names
            )
            
        raise NotImplementedError(f"Physical translation for {type(logical_plan).__name__} not implemented")
