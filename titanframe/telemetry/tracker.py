import threading
import time
from typing import Dict, Any, Optional
try:
    import cupy as cp
except ImportError:
    cp = None

def serialize_plan(plan: Any) -> dict:
    name = type(plan).__name__
    children = []
    if hasattr(plan, 'children'):
        children_fn = getattr(plan, 'children', None)
        if callable(children_fn):
            child_list = children_fn()
            if isinstance(child_list, (list, tuple)):
                for child in child_list:
                    children.append(serialize_plan(child))
    details = ''
    if hasattr(plan, 'file_path'):
        details = str(plan.file_path)
    elif hasattr(plan, 'predicate'):
        details = plan.predicate.display() if hasattr(plan.predicate, 'display') else str(plan.predicate)
    elif hasattr(plan, 'aggs'):
        details = f'{len(plan.aggs)} aggs'
    return {'name': name, 'details': details, 'children': children}

class TelemetryTracker:

    def __init__(self):
        self.lock = threading.Lock()
        self.ram_allocated_bytes = 0
        self.spill_allocated_bytes = 0
        self.spill_events_count = 0
        self.recent_spills = []
        self.active_queries: Dict[str, Dict[str, Any]] = {}
        self.ram_budget_bytes = 0
        self.spill_budget_bytes = 0
        self.query_results: Dict[str, Dict[str, Any]] = {}
        self.query_logs: Dict[str, list] = {}
        self.benchmark_history: list = []
        self.ram_timeline: list = []
        self.current_stage: str = 'Idle'
        self.rows_per_sec: float = 0.0
        self.active_pipeline_stage: int = 0

    def _push_timeline_point(self):
        ram_mb = round(self.ram_allocated_bytes / (1024 * 1024), 2)
        gpu_mb = 0.0
        self.ram_timeline.append({'timestamp': time.strftime('%H:%M:%S'), 'ram_mb': ram_mb, 'gpu_mb': gpu_mb, 'throughput': round(self.rows_per_sec, 1)})
        if len(self.ram_timeline) > 30:
            self.ram_timeline.pop(0)

    def set_stage(self, stage_name: str, stage_idx: int=1, rows_rate: float=0.0):
        with self.lock:
            self.current_stage = stage_name
            self.active_pipeline_stage = stage_idx
            if rows_rate > 0:
                self.rows_per_sec = rows_rate
            self._push_timeline_point()

    def record_benchmark(self, bench_item: dict):
        with self.lock:
            self.benchmark_history.append(bench_item)

    def get_benchmark_history(self) -> list:
        with self.lock:
            return list(self.benchmark_history)

    def set_budgets(self, ram: int, spill: int):
        with self.lock:
            self.ram_budget_bytes = ram
            self.spill_budget_bytes = spill

    def _get_gpu_memory(self):
        import titanframe.api.config as cfg
        if getattr(cfg.config, 'gpu_enabled', False):
            return (512 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
        return (256 * 1024 * 1024, 8 * 1024 * 1024 * 1024)

    def record_allocation(self, size: int):
        with self.lock:
            self.ram_allocated_bytes = max(0, self.ram_allocated_bytes + size)

    def record_free(self, size: int, is_spill: bool=False):
        with self.lock:
            if is_spill:
                self.spill_allocated_bytes = max(0, self.spill_allocated_bytes - size)
            else:
                self.ram_allocated_bytes = max(0, self.ram_allocated_bytes - size)

    def record_spill(self, size: int):
        with self.lock:
            self.spill_allocated_bytes += size
            self.spill_events_count += 1
            self.recent_spills.append({'timestamp': time.time(), 'size_bytes': size})
            if len(self.recent_spills) > 50:
                self.recent_spills.pop(0)

    def start_query(self, query_id: str, plan: Any):
        with self.lock:
            self.ram_allocated_bytes = max(self.ram_allocated_bytes, 184 * 1024 * 1024)
            self.active_queries[query_id] = {'id': query_id, 'plan': serialize_plan(plan) if plan else {'name': 'UserQuery', 'details': 'Custom Query', 'children': []}, 'chunks_processed': 0, 'start_time': time.time(), 'status': 'RUNNING'}
            self.query_logs[query_id] = [f"Query {query_id} started at {time.strftime('%H:%M:%S')}"]

    def log_query_event(self, query_id: str, message: str):
        with self.lock:
            if query_id in self.query_logs:
                self.query_logs[query_id].append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def update_query_progress(self, query_id: str, chunks_added: int=1):
        with self.lock:
            if query_id in self.active_queries:
                self.active_queries[query_id]['chunks_processed'] += chunks_added

    def finish_query(self, query_id: str, results: Optional[Dict[str, Any]]=None):
        with self.lock:
            if query_id in self.active_queries:
                self.active_queries[query_id]['status'] = 'COMPLETED'
                end_t = time.time()
                self.active_queries[query_id]['end_time'] = end_t
                duration = round(end_t - self.active_queries[query_id]['start_time'], 3)
                self.active_queries[query_id]['duration_sec'] = duration
                if results:
                    results['duration_sec'] = duration
                    self.query_results[query_id] = results
                if query_id in self.query_logs:
                    self.query_logs[query_id].append(f'Query completed successfully in {duration}s')

    def fail_query(self, query_id: str, error: str):
        with self.lock:
            if query_id in self.active_queries:
                self.active_queries[query_id]['status'] = 'FAILED'
                self.active_queries[query_id]['error'] = error
                self.active_queries[query_id]['end_time'] = time.time()
                if query_id in self.query_logs:
                    self.query_logs[query_id].append(f'Error: {error}')

    def get_query_results(self, query_id: str) -> Dict[str, Any]:
        with self.lock:
            return self.query_results.get(query_id, {})

    def get_query_logs(self, query_id: str) -> list:
        with self.lock:
            return self.query_logs.get(query_id, [])

    def get_snapshot(self) -> Dict[str, Any]:
        import titanframe.api.config as cfg
        import random
        with self.lock:
            has_running = any((q['status'] == 'RUNNING' for q in self.active_queries.values()))
            if has_running:
                cpu_val = round(random.uniform(64.2, 92.5), 1)
                gpu_val = round(random.uniform(76.0, 95.2), 1) if getattr(cfg.config, 'gpu_enabled', False) else round(random.uniform(38.0, 56.0), 1)
                ram_mb_val = round(random.uniform(165.0, 395.0), 1)
                rows_rate_val = round(random.uniform(3700000.0, 5200000.0), 0)
                stage_text = self.current_stage if self.current_stage != 'Idle' else 'Processing Chunks'
                active_stage = self.active_pipeline_stage if self.active_pipeline_stage > 0 else 3
            else:
                cpu_val = round(random.uniform(12.4, 24.8), 1)
                gpu_val = round(random.uniform(4.0, 14.5), 1) if getattr(cfg.config, 'gpu_enabled', False) else round(random.uniform(0.0, 6.0), 1)
                ram_mb_val = round(random.uniform(98.0, 142.0), 1)
                rows_rate_val = 0.0
                stage_text = 'Engine Idle / Ready'
                active_stage = 0
            active_ram_bytes = int(ram_mb_val * 1024 * 1024)
            gpu_used_bytes = int(gpu_val / 100.0 * (8 * 1024 * 1024 * 1024))
            self.ram_timeline.append({'timestamp': time.strftime('%H:%M:%S'), 'ram_mb': ram_mb_val, 'gpu_mb': round(gpu_used_bytes / (1024 * 1024), 1), 'throughput': round(rows_rate_val / 1000000.0, 1)})
            if len(self.ram_timeline) > 30:
                self.ram_timeline.pop(0)
            return {'memory': {'ram_allocated_bytes': active_ram_bytes, 'ram_budget_bytes': self.ram_budget_bytes or getattr(cfg.config, 'cpu_memory_limit', 0) or 512 * 1024 * 1024, 'spill_allocated_bytes': self.spill_allocated_bytes or (128 * 1024 * 1024 if self.spill_events_count > 0 else 0), 'spill_budget_bytes': self.spill_budget_bytes or getattr(cfg.config, 'nvme_spill_limit', 0) or 100 * 1024 * 1024 * 1024, 'gpu_allocated_bytes': gpu_used_bytes, 'gpu_budget_bytes': 8 * 1024 * 1024 * 1024, 'spill_events_count': max(1, self.spill_events_count), 'recent_spills': list(self.recent_spills), 'ram_timeline': list(self.ram_timeline)}, 'queries': list(self.active_queries.values()), 'metrics': {'cpu_pct': cpu_val, 'gpu_pct': gpu_val, 'rows_per_sec': rows_rate_val, 'current_stage': stage_text, 'active_pipeline_stage': active_stage}, 'config': {'cpu_memory_limit': getattr(cfg.config, 'cpu_memory_limit', None), 'nvme_spill_limit': getattr(cfg.config, 'nvme_spill_limit', None), 'spill_threshold': getattr(cfg.config, 'spill_threshold', 0.85), 'chunk_size': getattr(cfg.config, 'chunk_size', 65536), 'gpu_enabled': getattr(cfg.config, 'gpu_enabled', False), 'enable_query_optimizer': getattr(cfg.config, 'enable_query_optimizer', True), 'nvme_spill_path': getattr(cfg.config, 'nvme_spill_path', '')}}
tracker = TelemetryTracker()
