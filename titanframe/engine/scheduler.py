import pyarrow as pa
from titanframe.plan.physical.node import PhysicalPlan, ExecutionContext

class DAGScheduler:

    def execute(self, plan: PhysicalPlan, ctx: ExecutionContext, show_progress: bool=True) -> pa.Table:
        import pyarrow as pa
        from titanframe.telemetry.tracker import tracker
        import uuid
        query_id = str(uuid.uuid4())[:8]
        tracker.start_query(query_id, plan)
        batches = []
        iterator = plan.execute(ctx)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc='Executing Plan', unit='chunk', leave=False)
        try:
            for chunk in iterator:
                tracker.update_query_progress(query_id, 1)
                batches.append(chunk.data)
            tracker.finish_query(query_id)
        except Exception as e:
            tracker.fail_query(query_id, str(e))
            raise e
        if not batches:
            return None
        return pa.Table.from_batches(batches)
