import React, { useEffect, useState, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import axios from 'axios';

const GraphCanvas = ({ onNodeClick, controlParams }) => {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [hoverNode, setHoverNode] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const fgRef = useRef();

  const fetchData = useCallback(async () => {
    try {
      // Fetch nodes and edges in parallel
      const [nodesRes, edgesRes] = await Promise.all([
        axios.get('http://localhost:8000/v1/nodes?limit=1000'),
        axios.get('http://localhost:8000/v1/edges?limit=2000')
      ]);

      const nodes = nodesRes.data.map((node, i) => ({
        id: node.uuid,
        ...node,
        val: 1,
        // Random initial positions to prevent clustering
        x: (Math.random() - 0.5) * 1000,
        y: (Math.random() - 0.5) * 1000
      }));

      const links = edgesRes.data.map(edge => ({
        source: edge.source_id,
        target: edge.target_id,
        ...edge
      }));

      console.log(`Loaded ${nodes.length} nodes and ${links.length} links`);
      setGraphData({ nodes, links });
      
      // Force reconfiguration after data loads
      setTimeout(() => {
        if (fgRef.current) {
          // Much stronger repulsion for 500 nodes
          fgRef.current.d3Force('charge').strength(-500).distanceMax(500);
          fgRef.current.d3Force('link').distance(150).strength(0.1);
          fgRef.current.d3ReheatSimulation();
        }
      }, 100);
    } catch (error) {
      console.error("Error fetching graph data:", error);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    if (fgRef.current && graphData.nodes.length > 0) {
      // Configure forces with stronger repulsion
      fgRef.current.d3Force('charge').strength(-controlParams.chargeStrength).distanceMax(500);
      fgRef.current.d3Force('link').distance(controlParams.linkDistance).strength(0.1);
      
      // Reheat the simulation to apply new forces
      fgRef.current.d3ReheatSimulation();
    }
  }, [controlParams, graphData]);

  return (
    <div className="w-full h-full bg-slate-900">
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        nodeLabel="text"
        nodeColor={node => node.color || '#60a5fa'}
        nodeRelSize={controlParams.nodeSize}
        linkColor={() => '#94a3b8'}
        linkWidth={controlParams.linkWidth}
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={1}
        
        // Force configuration
        enableNodeDrag={true}
        enableZoomInteraction={true}
        enablePanInteraction={true}
        
        // Initial zoom out so graph isn't cramped
        zoom={0.5}
        minZoom={0.1}
        maxZoom={8}
        
        // Custom rendering for edge labels
        linkCanvasObject={(link, ctx, globalScale) => {
          // Draw the link line
          ctx.beginPath();
          ctx.moveTo(link.source.x, link.source.y);
          ctx.lineTo(link.target.x, link.target.y);
          ctx.strokeStyle = '#64748b'; // Slate-500 (brighter)
          ctx.lineWidth = Math.max(controlParams.linkWidth, 0.5); // Minimum width
          ctx.stroke();

          // Only draw label if hovered or very zoomed in
          if (globalScale > 1.5) { // Show sooner (was 2.5)
            const label = link.relation;
            if (!label) return;

            const start = link.source;
            const end = link.target;
            const textPos = {
              x: start.x + (end.x - start.x) / 2,
              y: start.y + (end.y - start.y) / 2
            };

            const fontSize = 6; // Increased from 3
            ctx.font = `${fontSize}px Sans-Serif`;
            const textWidth = ctx.measureText(label).width;
            const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.4); // More padding

            ctx.save();
            ctx.translate(textPos.x, textPos.y);
            
            const relLink = { x: end.x - start.x, y: end.y - start.y };
            let textAngle = Math.atan2(relLink.y, relLink.x);
            if (textAngle > Math.PI / 2) textAngle = -(Math.PI - textAngle);
            if (textAngle < -Math.PI / 2) textAngle = -(-Math.PI - textAngle);
            ctx.rotate(textAngle);

            ctx.fillStyle = 'rgba(15, 23, 42, 0.9)'; // Darker background
            ctx.beginPath();
            ctx.roundRect(-bckgDimensions[0] / 2, -bckgDimensions[1] / 2, bckgDimensions[0], bckgDimensions[1], 2);
            ctx.fill();

            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#e2e8f0'; // Lighter text
            ctx.fillText(label, 0, 0);
            ctx.restore();
          }
        }}
        linkCanvasObjectMode={() => 'replace'}

        // Custom rendering for nodes
        nodeCanvasObject={(node, ctx, globalScale) => {
          const isSelected = selectedNode && selectedNode.id === node.id;
          const isHovered = hoverNode && hoverNode.id === node.id;
          
          // Truncate label
          let label = node.text || '';
          if (label.length > 20 && !isHovered && !isSelected) {
            label = label.substring(0, 20) + '...';
          }

          const fontSize = (isSelected || isHovered ? 16 : 12) / globalScale; // Increased font size
          
          // Draw node circle
          ctx.beginPath();
          const size = controlParams.nodeSize * (isSelected || isHovered ? 1.5 : 1);
          ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
          ctx.fillStyle = node.color || (isSelected ? '#3b82f6' : '#60a5fa');
          ctx.fill();
          
          // Draw ring for selected/hovered
          if (isSelected || isHovered) {
            ctx.beginPath();
            ctx.arc(node.x, node.y, size + 2/globalScale, 0, 2 * Math.PI, false);
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1/globalScale;
            ctx.stroke();
          }

          // Draw label only if zoomed in, hovered, or selected
          if (globalScale > 1.2 || isHovered || isSelected) { // Show sooner (was 1.5)
            ctx.font = `${fontSize}px Sans-Serif`;
            const textWidth = ctx.measureText(label).width;
            
            // Text background
            ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
            ctx.roundRect(node.x + size + 2, node.y - fontSize/2 - 1, textWidth + 4, fontSize + 2, 2);
            ctx.fill();

            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.fillText(label, node.x + size + 4, node.y + fontSize/3);
          }
        }}
        nodeCanvasObjectMode={() => 'replace'}

        onNodeClick={(node) => {
          fgRef.current.centerAt(node.x, node.y, 1000);
          fgRef.current.zoom(2, 2000);
          onNodeClick(node);
          setSelectedNode(node);
        }}
        onNodeHover={(node) => setHoverNode(node)}
        
        // Force Engine Configuration for better structure
        d3VelocityDecay={0.4} // Higher friction for better control
        d3AlphaDecay={0.0228} // Standard cooling rate
        warmupTicks={100} // Pre-calculate some positions
        cooldownTicks={0} // Don't auto-stop
        
        backgroundColor="#0f172a"
      />
    </div>
  );
};

export default GraphCanvas;
