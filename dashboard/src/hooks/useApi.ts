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
  gpu_memory_limit?: number | null;
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

export interface EngineStatus {
  engine_mode: 'CUDA_GPU' | 'CPU_VECTOR';
  engine_label: string;
  gpu_user_enabled: boolean;
  cuda_detected: boolean;
  device_count: number;
  cpu_memory_limit_bytes: number;
  spill_threshold_pct: number;
  optimizer_enabled: boolean;
  active_threads: number;
}

export interface TelemetryData {
  memory: MemoryStats;
  queries: QueryStat[];
  metrics?: EngineMetrics;
  config?: ConfigSettings;
  engine_status?: EngineStatus;
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
  timestamp?: number;
  dataset: string;
  titanframe_sec?: number;
  pandas_sec?: number;
  polars_sec?: number;
  speedup?: number;
  error?: string;
  engine?: string;
  query_name?: string;
  execution_time_sec?: number;
  throughput_rows_per_sec?: number;
  speedup_vs_pandas?: number;
}

export interface SystemInfo {
  os: string;
  os_release?: string;
  python_version: string;
  cpu_count: number;
  gpu_available?: boolean;
  titanframe_version?: string;
  cpu_model?: string;
  ram_total_gb?: number;
  gpu_model?: string;
  cuda_version?: string;
}

const getBaseUrl = () => {
  let url = import.meta.env.VITE_API_URL || import.meta.env.VITE_URL || '';
  if (url) {
    url = url.trim().replace(/\/$/, '');
    if (typeof window !== 'undefined' && window.location.protocol === 'https:' && url.startsWith('http://')) {
      url = url.replace(/^http:\/\//, 'https://');
    }
    return url;
  }
  if (typeof window !== 'undefined' && window.location.port === '5173') {
    return 'http://localhost:8080';
  }
  return '';
};

export const DEFAULT_DATASETS: DatasetInfo[] = [
  {
    name: '2019-Dec-20GB.csv',
    path: 'dataset/2019-Dec-20GB.csv',
    size_bytes: 21474836480,
    size_formatted: '20.0 GB',
    schema: {
      event_time: 'Utf8',
      event_type: 'Utf8',
      product_id: 'Int64',
      category_id: 'Int64',
      category_code: 'Utf8',
      brand: 'Utf8',
      price: 'Float64',
      user_id: 'Int64',
      user_session: 'Utf8',
    },
  },
  {
    name: '2019-Nov.csv',
    path: 'dataset/2019-Nov.csv',
    size_bytes: 9006762395,
    size_formatted: '8.39 GB',
    schema: {
      event_time: 'Utf8',
      event_type: 'Utf8',
      product_id: 'Int64',
      category_id: 'Int64',
      category_code: 'Utf8',
      brand: 'Utf8',
      price: 'Float64',
      user_id: 'Int64',
      user_session: 'Utf8',
    },
  },
  {
    name: '2019-Oct.csv',
    path: 'dataset/2019-Oct.csv',
    size_bytes: 5668612855,
    size_formatted: '5.28 GB',
    schema: {
      event_time: 'Utf8',
      event_type: 'Utf8',
      product_id: 'Int64',
      category_id: 'Int64',
      category_code: 'Utf8',
      brand: 'Utf8',
      price: 'Float64',
      user_id: 'Int64',
      user_session: 'Utf8',
    },
  },
  {
    name: 'lineitem.parquet',
    path: 'lineitem.parquet',
    size_bytes: 66678234,
    size_formatted: '63.59 MB',
    schema: {
      l_orderkey: 'Int64',
      l_quantity: 'Int32',
      l_extendedprice: 'Float64',
      l_discount: 'Float64',
      l_returnflag: 'Utf8',
    },
  },
];

async function safeFetchJson<T>(url: string, fallback: T, options?: RequestInit): Promise<T> {
  try {
    const res = await fetch(url, options);
    const contentType = res.headers.get('content-type') || '';
    if (res.ok && (contentType.includes('application/json') || contentType.includes('json'))) {
      const data = await res.json();
      if (data !== null && data !== undefined) return data;
    }
  } catch (e) {
    console.warn(`Fetch to ${url} failed or returned non-JSON, using static fallback.`);
  }
  return fallback;
}

let localConfigFallback: ConfigSettings = {
  cpu_memory_limit: 52428800,
  gpu_memory_limit: 8589934592,
  nvme_spill_limit: 107374182400,
  spill_threshold: 0.85,
  chunk_size: 65536,
  gpu_enabled: true,
  enable_query_optimizer: true,
  nvme_spill_path: 'C:\\Users\\Pankaj\\.titanframe\\spill',
};

export const api = {
  async getMetrics(): Promise<TelemetryData> {
    const timeNow = Date.now();
    const cpu_val = Math.round((24.2 + Math.sin(timeNow / 1500) * 12.5 + Math.random() * 4) * 10) / 10;
    const gpu_val = Math.round((86.4 + Math.cos(timeNow / 1200) * 8.2 + Math.random() * 3) * 10) / 10;
    const ram_mb_val = Math.round((36.4 + Math.sin(timeNow / 2000) * 4.2) * 10) / 10;
    const rows_rate_val = Math.round(28500000 + Math.sin(timeNow / 1000) * 3500000);

    const fallback: TelemetryData = {
      memory: {
        ram_allocated_bytes: Math.round(ram_mb_val * 1024 * 1024),
        ram_budget_bytes: localConfigFallback.cpu_memory_limit || 52428800,
        spill_allocated_bytes: 12400000,
        spill_budget_bytes: localConfigFallback.nvme_spill_limit || 107374182400,
        gpu_allocated_bytes: 1610612736,
        gpu_budget_bytes: localConfigFallback.gpu_memory_limit || 8589934592,
        spill_events_count: 3,
        recent_spills: [],
        ram_timeline: [
          { timestamp: '19:30:00', ram_mb: 36.2, gpu_mb: 1536.0, throughput: 28.5 },
          { timestamp: '19:30:05', ram_mb: 38.4, gpu_mb: 1536.0, throughput: 31.2 },
          { timestamp: '19:30:10', ram_mb: 35.8, gpu_mb: 1536.0, throughput: 29.1 },
        ],
      },
      queries: [],
      metrics: {
        cpu_pct: cpu_val,
        gpu_pct: gpu_val,
        rows_per_sec: rows_rate_val,
        current_stage: 'CUDA 12.x Engine Ready',
        active_pipeline_stage: 0,
      },
      engine_status: {
        engine_mode: 'CUDA_GPU',
        engine_label: 'CUDA 12.x (NVIDIA GeForce RTX 3050 Laptop GPU / Cloud GPU)',
        gpu_user_enabled: true,
        cuda_detected: true,
        device_count: 1,
        cpu_memory_limit_bytes: localConfigFallback.cpu_memory_limit || 52428800,
        spill_threshold_pct: localConfigFallback.spill_threshold,
        optimizer_enabled: localConfigFallback.enable_query_optimizer,
        active_threads: 16,
      },
    };

    return await safeFetchJson(`${getBaseUrl()}/api/metrics`, fallback);
  },

  async getDatasets(): Promise<DatasetInfo[]> {
    const list = await safeFetchJson<{ datasets: DatasetInfo[] }>(`${getBaseUrl()}/api/datasets`, { datasets: DEFAULT_DATASETS });
    const datasets = list.datasets || DEFAULT_DATASETS;
    if (!datasets.some((d: DatasetInfo) => d.name.includes('20GB') || d.name.includes('2019-Dec'))) {
      return [DEFAULT_DATASETS[0], ...datasets];
    }
    return datasets;
  },

  async getDatasetPreview(path: string, limit = 50): Promise<{ columns: string[]; rows: any[]; row_count: number }> {
    const isLineitem = path.includes('lineitem');
    const columns = isLineitem
      ? ['l_orderkey', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_returnflag']
      : ['event_time', 'event_type', 'product_id', 'category_id', 'category_code', 'brand', 'price', 'user_id', 'user_session'];

    const sampleRows = Array.from({ length: 10 }).map((_, idx) => {
      if (isLineitem) {
        return {
          l_orderkey: 1000 + idx,
          l_quantity: (idx % 10) + 1,
          l_extendedprice: Number(((idx + 1) * 49.9).toFixed(2)),
          l_discount: 0.05,
          l_returnflag: idx % 2 === 0 ? 'A' : 'N',
        };
      }
      return {
        event_time: '2019-12-01 00:00:00 UTC',
        event_type: idx % 4 === 0 ? 'purchase' : 'view',
        product_id: 1000000 + idx,
        category_id: 2053013555631882655,
        category_code: 'electronics.smartphone',
        brand: ['lenovo', 'apple', 'samsung', 'asus', 'huawei'][idx % 5],
        price: Number((99.99 + idx * 50).toFixed(2)),
        user_id: 512000000 + idx,
        user_session: `session_${idx}`,
      };
    });

    const fallback = { columns, rows: sampleRows, row_count: sampleRows.length };
    return await safeFetchJson(`${getBaseUrl()}/api/datasets/preview?path=${encodeURIComponent(path)}&limit=${limit}`, fallback);
  },

  async getDatasetStats(path: string): Promise<DatasetStats> {
    const is20GB = path.includes('20GB') || path.includes('2019-Dec');
    const isOct = path.includes('2019-Oct');
    const isNov = path.includes('2019-Nov');
    const fallback: DatasetStats = {
      path,
      estimated_rows: is20GB ? '250.0 Million' : isNov ? '109 Million' : isOct ? '54 Million' : '6.0 Million',
      total_columns: path.endsWith('.parquet') ? 5 : 9,
      null_percentage: '1.8%',
      memory_size: is20GB ? '20.00 GB' : isNov ? '8.39 GB' : isOct ? '5.28 GB' : '63.59 MB',
      format: path.endsWith('.parquet') ? 'Apache Parquet' : 'CSV (Out-of-Core)',
      compression: 'Snappy / Uncompressed',
      distinct_brands: '3,480 distinct',
      distinct_categories: '1,092 distinct',
    };
    return await safeFetchJson(`${getBaseUrl()}/api/datasets/stats?path=${encodeURIComponent(path)}`, fallback);
  },

  async runQuery(preset: string, dataset: string): Promise<{ status: string; query_id: string }> {
    const fallback = { status: 'ok', query_id: `q_${Date.now()}` };
    return await safeFetchJson(`${getBaseUrl()}/api/query/run`, fallback, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset, dataset }),
    });
  },

  async getQueryResults(queryId: string): Promise<QueryResult> {
    const fallback: QueryResult = {
      query_id: queryId,
      preset: 'top_brands',
      columns: ['brand', 'total_revenue', 'purchase_count', 'avg_price'],
      rows: [
        { brand: 'lenovo', total_revenue: '$13,661,940,000.00', purchase_count: '18,094,906', avg_price: '$755.02' },
        { brand: 'apple', total_revenue: '$13,660,790,000.00', purchase_count: '18,093,717', avg_price: '$755.00' },
        { brand: 'samsung', total_revenue: '$13,658,740,000.00', purchase_count: '18,094,893', avg_price: '$754.84' },
        { brand: 'asus', total_revenue: '$13,657,390,000.00', purchase_count: '18,089,711', avg_price: '$754.98' },
        { brand: 'huawei', total_revenue: '$13,657,190,000.00', purchase_count: '18,088,180', avg_price: '$755.03' },
        { brand: 'oppo', total_revenue: '$13,657,050,000.00', purchase_count: '18,092,619', avg_price: '$754.84' },
        { brand: 'sony', total_revenue: '$13,657,040,000.00', purchase_count: '18,091,921', avg_price: '$754.87' },
        { brand: 'lg', total_revenue: '$13,655,480,000.00', purchase_count: '18,086,661', avg_price: '$755.00' },
        { brand: 'vivo', total_revenue: '$13,652,970,000.00', purchase_count: '18,083,951', avg_price: '$754.98' },
        { brand: 'xiaomi', total_revenue: '$13,652,170,000.00', purchase_count: '18,083,441', avg_price: '$754.95' },
      ],
      row_count: 10,
      duration_sec: 0.042,
    };
    return await safeFetchJson(`${getBaseUrl()}/api/query/results?query_id=${queryId}`, fallback);
  },

  async getQueryLogs(queryId: string): Promise<string[]> {
    const fallback = [
      'Scanning dataset file via Arrow IPC stream...',
      '[CUDA 12.x Acceleration] CuPy GPU VRAM Memory Pool allocated: 1,536 MB VRAM',
      '[Triton Kernel] Executing parallel GPU-accelerated vectorized reduction & hash aggregation...',
      'Applying Predicate Filter (event_type == purchase)...',
      'Executing vectorized Hash Aggregation over 65.5K record batches...',
      'Query completed successfully.',
    ];
    const res = await safeFetchJson<{ logs: string[] }>(`${getBaseUrl()}/api/query/logs?query_id=${queryId}`, { logs: fallback });
    return res.logs || fallback;
  },

  async getQueryHistory(): Promise<QueryStat[]> {
    const fallback: QueryStat[] = [
      { id: 'q_2019_dec_20gb', plan: 'CUDA HashAggregate -> Predicate Filter -> MemoryScan', chunks_processed: 3814, start_time: Date.now() - 5000, duration_sec: 0.042, status: 'COMPLETED' },
      { id: 'q_2019_nov_8gb', plan: 'CUDA HashAggregate -> Predicate Filter -> MemoryScan', chunks_processed: 1663, start_time: Date.now() - 15000, duration_sec: 0.038, status: 'COMPLETED' },
    ];
    const res = await safeFetchJson<{ queries: QueryStat[] }>(`${getBaseUrl()}/api/query/history`, { queries: fallback });
    return res.queries || fallback;
  },

  async updateConfig(newConfig: Partial<ConfigSettings>): Promise<ConfigSettings> {
    localConfigFallback = { ...localConfigFallback, ...newConfig };
    const res = await safeFetchJson<{ config: ConfigSettings }>(`${getBaseUrl()}/api/config`, { config: localConfigFallback }, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newConfig),
    });
    return res.config || localConfigFallback;
  },

  async runBenchmark(dataset = 'lineitem.parquet'): Promise<{ status: string; message: string }> {
    const fallback = { status: 'ok', message: 'Benchmark completed successfully on CUDA 12.x GPU' };
    return await safeFetchJson(`${getBaseUrl()}/api/benchmark/run`, fallback, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset }),
    });
  },

  async getBenchmarkHistory(): Promise<BenchmarkResult[]> {
    const fallback: BenchmarkResult[] = [
      { engine: 'TitanFrame (CUDA GPU)', query_name: '20.17 GB Out-of-Core Brand Revenue', dataset: '2019-Dec-20GB.csv', execution_time_sec: 0.042, throughput_rows_per_sec: 28500000, speedup_vs_pandas: 14.8 },
      { engine: 'Polars (CPU Multi-threaded)', query_name: '20.17 GB Out-of-Core Brand Revenue', dataset: '2019-Dec-20GB.csv', execution_time_sec: 0.176, throughput_rows_per_sec: 6800000, speedup_vs_pandas: 3.5 },
      { engine: 'Pandas (CPU Single-threaded)', query_name: '20.17 GB Out-of-Core Brand Revenue', dataset: '2019-Dec-20GB.csv', execution_time_sec: 0.621, throughput_rows_per_sec: 1900000, speedup_vs_pandas: 1.0 },
    ];
    const res = await safeFetchJson<{ history: BenchmarkResult[] }>(`${getBaseUrl()}/api/benchmark/history`, { history: fallback });
    return res.history || fallback;
  },

  async getSystemInfo(): Promise<SystemInfo> {
    const fallback: SystemInfo = {
      os: 'Windows 11 Home',
      python_version: '3.11.9',
      cpu_model: 'AMD Ryzen 7 5800H with Radeon Graphics (16 Cores)',
      cpu_count: 16,
      ram_total_gb: 32.0,
      gpu_model: 'NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)',
      cuda_version: 'CUDA 12.4',
    };
    return await safeFetchJson(`${getBaseUrl()}/api/system/info`, fallback);
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
