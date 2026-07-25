import { useState, useEffect } from 'react';
import './index.css';
import './App.css';
import { useTelemetry, api } from './hooks/useApi';
import type { DatasetInfo } from './hooks/useApi';
import { Sidebar } from './components/Sidebar';
import type { PageId } from './components/Sidebar';
import { StatusBar } from './components/StatusBar';



import { ExecutiveDashboard } from './pages/ExecutiveDashboard';
import { DatasetExplorer } from './pages/DatasetExplorer';
import { SqlWorkspace } from './pages/SqlWorkspace';
import { QueryPlanVisualizer } from './pages/QueryPlanVisualizer';
import { BenchmarkDashboard } from './pages/BenchmarkDashboard';
import { MemoryMonitor } from './pages/MemoryMonitor';
import { QueryHistory } from './pages/QueryHistory';
import { Settings } from './pages/Settings';

const DEFAULT_DATASETS: DatasetInfo[] = [
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

export default function App() {
  const [activePage, setActivePage] = useState<PageId>('executive');
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);
  const [refreshInterval, setRefreshInterval] = useState<number>(1000);
  const [datasets, setDatasets] = useState<DatasetInfo[]>(DEFAULT_DATASETS);
  const [selectedDatasetForQuery, setSelectedDatasetForQuery] = useState<string | undefined>();

  const { telemetry, isConnected } = useTelemetry(refreshInterval);

  useEffect(() => {
    api
      .getDatasets()
      .then((data) => {
        if (data && data.length > 0) {
          setDatasets(data);
        }
      })
      .catch(console.error);
  }, []);

  const handleSelectDatasetForQuery = (path: string) => {
    setSelectedDatasetForQuery(path);
    setActivePage('sql');
  };

  const activeQueriesCount =
    telemetry?.queries.filter((q) => q.status === 'RUNNING').length || 0;

  return (
    <div className="titan-app-shell">
      <div className="shell-body">
        {}
        <Sidebar
          activePage={activePage}
          onSelectPage={setActivePage}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
          activeQueriesCount={activeQueriesCount}
        />

        {}
        <main className="shell-content">
          {activePage === 'executive' && (
            <ExecutiveDashboard
              telemetry={telemetry}
              datasets={datasets}
              onNavigate={setActivePage}
            />
          )}

          {activePage === 'datasets' && (
            <DatasetExplorer
              datasets={datasets}
              onSelectForQuery={handleSelectDatasetForQuery}
            />
          )}

          {activePage === 'sql' && (
            <SqlWorkspace
              datasets={datasets}
              selectedDatasetPath={selectedDatasetForQuery}
              onNavigateToDag={() => setActivePage('visualizer')}
              telemetry={telemetry}
            />
          )}

          {activePage === 'visualizer' && (
            <QueryPlanVisualizer telemetry={telemetry} />
          )}

          {activePage === 'benchmarks' && (
            <BenchmarkDashboard datasets={datasets} />
          )}

          {activePage === 'memory' && (
            <MemoryMonitor telemetry={telemetry} />
          )}

          {activePage === 'history' && (
            <QueryHistory telemetry={telemetry} />
          )}

          {activePage === 'settings' && (
            <Settings telemetry={telemetry} />
          )}
        </main>
      </div>

      {}
      <StatusBar
        isConnected={isConnected}
        telemetry={telemetry}
        refreshInterval={refreshInterval}
        onRefreshIntervalChange={setRefreshInterval}
      />
    </div>
  );
}
