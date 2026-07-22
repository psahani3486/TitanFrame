import React from 'react';
import './LivePipelineAnimation.css';

interface LivePipelineAnimationProps {
  activeStageIdx: number; // 0: Idle, 1: Scan, 2: Projection, 3: Filter, 4: Agg, 5: Output
  currentStageName: string;
  rowsPerSec: number;
}

const PIPELINE_NODES = [
  { id: 1, label: 'Read Dataset', desc: 'Scan Arrow Chunks' },
  { id: 2, label: 'Projection', desc: 'Prune Unused Cols' },
  { id: 3, label: 'Predicate Filter', desc: 'Vector Evaluation' },
  { id: 4, label: 'Hash Aggregation', desc: 'Group & Compute' },
  { id: 5, label: 'Output Sink', desc: 'Final Row Set' },
];

export const LivePipelineAnimation: React.FC<LivePipelineAnimationProps> = ({
  activeStageIdx,
  currentStageName,
  rowsPerSec,
}) => {
  const isExecuting = activeStageIdx > 0 && activeStageIdx <= 5;

  return (
    <div className="pipeline-container glass-panel">
      <div className="pipeline-header">
        <div>
          <h4 className="pipeline-title">Live Query Streaming Pipeline</h4>
          <span className="pipeline-sub">
            Stage: <strong>{currentStageName}</strong>
            {rowsPerSec > 0 && ` | Throughput: ${(rowsPerSec / 1000000).toFixed(2)}M rows/s`}
          </span>
        </div>
        <div className={`pipeline-status-badge ${isExecuting ? 'running' : 'idle'}`}>
          <span className="pulse-dot"></span>
          <span>{isExecuting ? 'PROCESSING CHUNKS' : 'ENGINE READY'}</span>
        </div>
      </div>

      <div className="pipeline-nodes-row">
        {PIPELINE_NODES.map((node, index) => {
          const isActive = activeStageIdx === node.id;
          const isPassed = activeStageIdx > node.id || activeStageIdx === 5;

          return (
            <React.Fragment key={node.id}>
              <div
                className={`pipeline-node ${isActive ? 'active' : ''} ${
                  isPassed ? 'passed' : ''
                }`}
              >
                <div className="node-step">{node.id}</div>
                <div className="node-text-group">
                  <span className="node-label">{node.label}</span>
                  <span className="node-desc">{node.desc}</span>
                </div>
              </div>

              {index < PIPELINE_NODES.length - 1 && (
                <div className={`pipeline-connector ${isPassed ? 'active' : ''}`}>
                  <div className={`particle-emitter ${isExecuting ? 'animating' : ''}`}></div>
                  <div className="connector-line"></div>
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};
