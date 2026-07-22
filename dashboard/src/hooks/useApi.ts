import { useState, useEffect } from 'react';

export interface PlanNode {
  name: string;
  details: string;
  children: PlanNode[];
}

export interface MemoryStats {
  ram_allocated_bytes: number;
  ram_budget_bytes: number;
  spill_allocated_bytes: number;
  spill_budget_bytes: number;
  gpu_allocated_bytes: number;
  gpu_budget_bytes: number;
  spill_events_count: number;
  recent_spills: Array<{ timestamp: number; size_bytes: number }>;
  ram_timeline?: Array<{ timestamp: string; ram_mb: number; gpu_mb: number; throughput: number }>;
}

export interface QueryStat {
  id: string;
  plan: PlanNode | string;
  chunks_processed: number;
  start_time: number;
  end_time?: number;
  duration_sec?: number;
  status: 'RUNNING' | 'COMPLETED' | 'FAILED';
  error?: string;
}

export interface EngineMetrics {
  cpu_pct: number;
  gpu_pct: number;
  rows_per_sec: number;
  current_stage: string;
  active_pipeline_stage: number;
}

export interface ConfigSettings {
  cpu_memory_limit: number | null;
  nvme_spill_limit: number | null;
  spill_threshold: number;
  chunk_size: number;
  gpu_enabled: boolean;
  enable_query_optimizer: boolean;
  nvme_spill_path: string;

  num_threads?: number;
  simd_enabled?: boolean;
  arrow_buffer_kb?: number;
  compression_mode?: string;
  scheduler_policy?: string;
}

export interface TelemetryData {
  memory: MemoryStats;
  queries: QueryStat[];
  metrics?: EngineMetrics;
  config?: ConfigSettings;
}

export interface DatasetInfo {
  name: string;
  path: string;
  size_bytes: number;
  size_formatted: string;
  schema: Record<string, string>;
}

export interface DatasetStats {
  path: string;
  estimated_rows: string;
  total_columns: number;
  null_percentage: string;
  memory_size: string;
  format: string;
  compression: string;
  distinct_brands: string;
  distinct_categories: string;
}

export interface QueryResult {
  query_id: string;
  preset: string;
  columns: string[];
  rows: Record<string, any>[];
  row_count: number;
  duration_sec?: number;
}

export interface BenchmarkResult {
  timestamp: number;
  dataset: string;
  titanframe_sec?: number;
  pandas_sec?: number;
  polars_sec?: number;
  speedup?: number;
  error?: string;
}

export interface SystemInfo {
  os: string;
  os_release: string;
  python_version: string;
  cpu_count: number;
  gpu_available: boolean;
  titanframe_version: string;
}

const getBaseUrl = () => {
  if (typeof window !== 'undefined' && window.location.port === '5173') {
    return 'http://localhost:8080';
  }
  return '';
};

export const api = {
  async getMetrics(): Promise<TelemetryData> {
    const res = await fetch(`${getBaseUrl()}/api/metrics`);
    if (!res.ok) throw new Error('Failed to fetch metrics');
    return res.json();
  },

  async getDatasets(): Promise<DatasetInfo[]> {
    const res = await fetch(`${getBaseUrl()}/api/datasets`);
    if (!res.ok) throw new Error('Failed to fetch datasets');
    const json = await res.json();
    return json.datasets || [];
  },

  async getDatasetPreview(path: string, limit = 50): Promise<{ columns: string[]; rows: any[]; row_count: number }> {
    const res = await fetch(`${getBaseUrl()}/api/datasets/preview?path=${encodeURIComponent(path)}&limit=${limit}`);
    if (!res.ok) throw new Error('Failed to fetch dataset preview');
    return res.json();
  },

  async getDatasetStats(path: string): Promise<DatasetStats> {
    const res = await fetch(`${getBaseUrl()}/api/datasets/stats?path=${encodeURIComponent(path)}`);
    if (!res.ok) throw new Error('Failed to fetch dataset statistics');
    return res.json();
  },

  async runQuery(preset: string, dataset: string): Promise<{ status: string; query_id: string }> {
    const res = await fetch(`${getBaseUrl()}/api/query/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset, dataset }),
    });
    if (!res.ok) throw new Error('Failed to submit query');
    return res.json();
  },

  async getQueryResults(queryId: string): Promise<QueryResult> {
    const res = await fetch(`${getBaseUrl()}/api/query/results?query_id=${queryId}`);
    if (!res.ok) throw new Error('Failed to fetch query results');
    return res.json();
  },

  async getQueryLogs(queryId: string): Promise<string[]> {
    const res = await fetch(`${getBaseUrl()}/api/query/logs?query_id=${queryId}`);
    if (!res.ok) throw new Error('Failed to fetch query logs');
    const json = await res.json();
    return json.logs || [];
  },

  async getQueryHistory(): Promise<QueryStat[]> {
    const res = await fetch(`${getBaseUrl()}/api/query/history`);
    if (!res.ok) throw new Error('Failed to fetch query history');
    const json = await res.json();
    return json.queries || [];
  },

  async updateConfig(newConfig: Partial<ConfigSettings>): Promise<ConfigSettings> {
    const res = await fetch(`${getBaseUrl()}/api/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newConfig),
    });
    if (!res.ok) throw new Error('Failed to update config');
    const json = await res.json();
    return json.config;
  },

  async runBenchmark(dataset = 'lineitem.parquet'): Promise<{ status: string; message: string }> {
    const res = await fetch(`${getBaseUrl()}/api/benchmark/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset }),
    });
    if (!res.ok) throw new Error('Failed to trigger benchmark');
    return res.json();
  },

  async getBenchmarkHistory(): Promise<BenchmarkResult[]> {
    const res = await fetch(`${getBaseUrl()}/api/benchmark/history`);
    if (!res.ok) throw new Error('Failed to fetch benchmark history');
    const json = await res.json();
    return json.history || [];
  },

  async getSystemInfo(): Promise<SystemInfo> {
    const res = await fetch(`${getBaseUrl()}/api/system/info`);
    if (!res.ok) throw new Error('Failed to fetch system info');
    return res.json();
  },
};

export function useTelemetry(intervalMs = 1000) {
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    const fetchTelemetry = async () => {
      try {
        const data = await api.getMetrics();
        if (isMounted) {
          setTelemetry(data);
          setIsConnected(true);
          setError(null);
        }
      } catch (err: any) {
        if (isMounted) {
          setIsConnected(false);
          setError(err.message || 'Connection offline');
        }
      }
    };

    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, intervalMs);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [intervalMs]);

  return { telemetry, isConnected, error };
}

export function formatBytes(bytes: number): string {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
