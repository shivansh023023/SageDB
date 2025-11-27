import React, { useEffect, useState, useRef } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import axios from 'axios';

const VectorSpace = ({ onNodeClick }) => {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const fgRef = useRef();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get('http://localhost:8000/v1/viz/vector-space');
        setGraphData({ nodes: res.data.nodes, links: [] });
      } catch (error) {
        console.error("Error fetching vector data:", error);
      }
    };
    fetchData();
  }, []);

  return (
    <div className="w-full h-full bg-slate-900">
      <ForceGraph3D
        ref={fgRef}
        graphData={graphData}
        nodeLabel="text"
        nodeColor={node => {
          // Color by type
          const type = node.group;
          if (type === 'document') return '#ef4444'; // Red
          if (type === 'chunk') return '#3b82f6'; // Blue
          return '#a855f7'; // Purple
        }}
        nodeVal={4}
        nodeResolution={16}
        showNavInfo={false}
        backgroundColor="#0f172a"
        onNodeClick={node => {
          // Aim at node from outside it
          const distance = 40;
          const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);

          fgRef.current.cameraPosition(
            { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }, // new position
            node, // lookAt ({ x, y, z })
            3000  // ms transition duration
          );
          
          onNodeClick(node);
        }}
      />
      <div className="absolute bottom-4 left-4 text-slate-500 text-xs pointer-events-none">
        <p>3D Vector Space (PCA Projection)</p>
        <p>Red: Documents | Blue: Chunks</p>
      </div>
    </div>
  );
};

export default VectorSpace;
