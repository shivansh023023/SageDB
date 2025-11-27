import networkx as nx
import os
import logging
import math
from typing import List, Dict, Optional
from config import GRAPH_PATH

logger = logging.getLogger(__name__)

class GraphManager:
    def __init__(self, graph_path: str = GRAPH_PATH):
        self.graph_path = graph_path
        self.graph = nx.DiGraph()

    def load_or_create(self):
        if os.path.exists(self.graph_path):
            logger.info(f"Loading graph from {self.graph_path}")
            try:
                self.graph = nx.read_gpickle(self.graph_path)
            except Exception as e:
                logger.error(f"Failed to load graph: {e}")
                self.graph = nx.DiGraph()
        else:
            logger.info("Creating new NetworkX graph")
            self.graph = nx.DiGraph()

    def add_node(self, uuid: str):
        self.graph.add_node(uuid)

    def remove_node(self, uuid: str):
        if self.graph.has_node(uuid):
            self.graph.remove_node(uuid)

    def add_edge(self, source: str, target: str, relation: str, weight: float):
        self.graph.add_edge(source, target, relation=relation, weight=weight)

    def save(self):
        nx.write_gpickle(self.graph, self.graph_path)

    def calculate_graph_score(self, candidate_node: str, seed_set: List[str]) -> float:
        """
        Calculate graph score based on connectivity to seeds and centrality.
        Formula: 0.7 * connectivity + 0.3 * centrality
        """
        if not self.graph.has_node(candidate_node):
            return 0.0

        # 1. Connectivity Score
        reachable_seeds = []
        min_distances = []
        
        for seed in seed_set:
            if seed == candidate_node:
                continue
            if self.graph.has_node(seed) and nx.has_path(self.graph, seed, candidate_node):
                try:
                    # Use weight for shortest path (smaller weight = closer?) 
                    # Note: Usually weights in graphs are distance (cost). 
                    # If weight is similarity (0-1), we might want 1/weight or 1-weight for distance.
                    # Prompt says: weight > 0 AND weight <= 1.0. 
                    # Assuming weight represents STRENGTH/SIMILARITY, so higher is better?
                    # Wait, prompt says: "min_distance = min(shortest_path_length...)"
                    # and "connectivity_score = exp(-min_distance)".
                    # If weight is strength, we should probably use 1/weight as distance.
                    # However, to stick strictly to prompt: "shortest_path_length(s, n, weight='weight')"
                    # This implies the stored 'weight' attribute is used as distance cost.
                    # So we assume the input weight IS the distance/cost.
                    
                    dist = nx.shortest_path_length(self.graph, seed, candidate_node, weight='weight')
                    reachable_seeds.append(seed)
                    min_distances.append(dist)
                except nx.NetworkXNoPath:
                    pass

        if min_distances:
            min_dist = min(min_distances)
            connectivity_score = math.exp(-min_dist)
        else:
            connectivity_score = 0.0

        # 2. Centrality Score (Degree Centrality Approximation)
        # Using in_degree + out_degree as per prompt
        degree = self.graph.in_degree(candidate_node) + self.graph.out_degree(candidate_node)
        
        # We need max_degree of the whole graph to normalize
        # Calculating max degree every time is expensive O(N). 
        # For a hackathon, maybe acceptable, or we can cache/estimate it.
        # Let's do it exact for now as graph size might not be huge.
        if len(self.graph) > 0:
            max_degree = max(d for n, d in self.graph.degree()) # degree() is in+out for DiGraph? No, degree is sum.
            # nx.degree returns (node, degree) iterator.
            # For DiGraph, degree = in_degree + out_degree.
            if max_degree > 0:
                centrality_score = degree / max_degree
            else:
                centrality_score = 0.0
        else:
            centrality_score = 0.0

        # Combined Score
        graph_score = 0.7 * connectivity_score + 0.3 * centrality_score
        return graph_score

    def get_bfs_subgraph(self, start_node: str, depth: int = 2, max_nodes: int = 50) -> Dict:
        if not self.graph.has_node(start_node):
            return {}
            
        # BFS traversal
        subgraph_nodes = set()
        queue = [(start_node, 0)]
        visited = {start_node}
        
        while queue and len(subgraph_nodes) < max_nodes:
            current, d = queue.pop(0)
            subgraph_nodes.add(current)
            
            if d < depth:
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))
        
        # Build response structure
        nodes = []
        edges = []
        
        for node in subgraph_nodes:
            nodes.append(node)
            for neighbor in self.graph.neighbors(node):
                if neighbor in subgraph_nodes:
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    edges.append({
                        "source": node,
                        "target": neighbor,
                        "relation": edge_data.get("relation"),
                        "weight": edge_data.get("weight")
                    })
                    
        return {"nodes": nodes, "edges": edges}

graph_manager = GraphManager()
