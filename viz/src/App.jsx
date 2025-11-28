import React, { useState } from 'react';
import GraphCanvas from './components/GraphCanvas';
import VectorSpace from './components/VectorSpace';
import ControlPanel from './components/ControlPanel';
import NodeDetails from './components/NodeDetails';
import { Network, Box } from 'lucide-react';

function App() {
  const [viewMode, setViewMode] = useState('graph'); // 'graph' or 'vector'
  const [selectedNode, setSelectedNode] = useState(null);
  const [controlParams, setControlParams] = useState({
    nodeSize: 3,
    linkWidth: 0.5,
    chargeStrength: 300,  // Higher repulsion to spread nodes
    linkDistance: 100     // Longer links
  });

  const handleNodeClick = (node) => {
    setSelectedNode(node);
  };

  const handleRefresh = () => {
    window.location.reload();
  };

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-slate-900 flex flex-col">
      {/* View Toggle */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 bg-slate-800/90 backdrop-blur rounded-full p-1 border border-slate-700 flex gap-1 shadow-xl">
        <button
          onClick={() => setViewMode('graph')}
          className={`px-4 py-1.5 rounded-full text-sm font-medium flex items-center gap-2 transition-all ${
            viewMode === 'graph' 
              ? 'bg-blue-500 text-white shadow-lg' 
              : 'text-slate-400 hover:text-white hover:bg-slate-700'
          }`}
        >
          <Network size={16} /> Graph
        </button>
        <button
          onClick={() => setViewMode('vector')}
          className={`px-4 py-1.5 rounded-full text-sm font-medium flex items-center gap-2 transition-all ${
            viewMode === 'vector' 
              ? 'bg-purple-500 text-white shadow-lg' 
              : 'text-slate-400 hover:text-white hover:bg-slate-700'
          }`}
        >
          <Box size={16} /> Vector
        </button>
      </div>

      {viewMode === 'graph' && (
        <ControlPanel 
          params={controlParams} 
          setParams={setControlParams} 
          onRefresh={handleRefresh}
        />
      )}
      
      <div className="flex-grow relative">
        {viewMode === 'graph' ? (
          <GraphCanvas 
            onNodeClick={handleNodeClick} 
            controlParams={controlParams}
          />
        ) : (
          <VectorSpace 
            onNodeClick={handleNodeClick}
          />
        )}
      </div>

      <NodeDetails 
        node={selectedNode} 
        onClose={() => setSelectedNode(null)} 
      />
    </div>
  );
}

export default App;
