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

    def remove_edge(self, source: str, target: str):
        """Remove edge between two nodes if it exists."""
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)

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

    def expand_from_seeds(self, seed_nodes: List[str], depth: int = 2) -> List[str]:
        """
        BFS expansion from seed nodes to discover related nodes.
        Returns a list of all nodes reachable within 'depth' hops from any seed.
        """
        expanded = set(seed_nodes)
        
        for seed in seed_nodes:
            if not self.graph.has_node(seed):
                continue
            try:
                # Get all nodes within 'depth' hops from this seed
                neighbors = nx.single_source_shortest_path_length(self.graph, seed, cutoff=depth)
                expanded.update(neighbors.keys())
                
                # Also check reverse direction (incoming edges)
                reverse_neighbors = nx.single_source_shortest_path_length(
                    self.graph.reverse(), seed, cutoff=depth
                )
                expanded.update(reverse_neighbors.keys())
            except Exception as e:
                logger.warning(f"Error expanding from seed {seed}: {e}")
                continue
        
        return list(expanded)

    def get_relationship_score(self, candidate: str, seed_nodes: List[str]) -> float:
        """
        Calculate a relationship-aware score based on edge types.
        Higher scores for semantically meaningful relationships.
        """
        # Edge type weights (more meaningful relationships get higher scores)
        edge_weights = {
            "is_a": 1.0,
            "part_of": 0.95,
            "specialization_of": 0.9,
            "uses": 0.85,
            "depends_on": 0.8,
            "implements": 0.8,
            "extends": 0.75,
            "related_to": 0.5,
            "mentioned_in": 0.3
        }
        
        scores = []
        
        for seed in seed_nodes:
            if seed == candidate:
                continue
                
            # Check direct edge from seed to candidate
            if self.graph.has_edge(seed, candidate):
                edge_data = self.graph.get_edge_data(seed, candidate)
                relation = edge_data.get('relation', 'related_to')
                weight = edge_weights.get(relation, 0.4)
                scores.append(weight)
            
            # Check direct edge from candidate to seed
            if self.graph.has_edge(candidate, seed):
                edge_data = self.graph.get_edge_data(candidate, seed)
                relation = edge_data.get('relation', 'related_to')
                weight = edge_weights.get(relation, 0.4)
                scores.append(weight)
        
        return max(scores) if scores else 0.0

    def calculate_expanded_graph_score(
        self, 
        candidate_node: str, 
        seed_set: List[str],
        seed_vector_scores: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive graph score for a candidate node.
        Returns breakdown of connectivity, centrality, and relationship scores.
        """
        if not self.graph.has_node(candidate_node):
            return {
                "connectivity": 0.0,
                "centrality": 0.0,
                "relationship": 0.0,
                "combined": 0.0
            }

        # 1. Connectivity Score (weighted by seed vector scores if available)
        distances = []
        weights = []
        
        for seed in seed_set:
            if seed == candidate_node:
                distances.append(0.0)
                weights.append(seed_vector_scores.get(seed, 1.0) if seed_vector_scores else 1.0)
                continue
            if not self.graph.has_node(seed):
                continue
                
            # Try both directions
            dist = float('inf')
            try:
                if nx.has_path(self.graph, seed, candidate_node):
                    dist = min(dist, nx.shortest_path_length(self.graph, seed, candidate_node))
            except:
                pass
            try:
                if nx.has_path(self.graph, candidate_node, seed):
                    dist = min(dist, nx.shortest_path_length(self.graph, candidate_node, seed))
            except:
                pass
                
            if dist != float('inf'):
                distances.append(dist)
                weights.append(seed_vector_scores.get(seed, 1.0) if seed_vector_scores else 1.0)

        if distances:
            # Weighted average distance (weighted by vector scores of seeds)
            if seed_vector_scores and sum(weights) > 0:
                weighted_avg_dist = sum(d * w for d, w in zip(distances, weights)) / sum(weights)
            else:
                weighted_avg_dist = sum(distances) / len(distances)
            connectivity_score = math.exp(-weighted_avg_dist)
        else:
            connectivity_score = 0.0

        # 2. Centrality Score
        degree = self.graph.in_degree(candidate_node) + self.graph.out_degree(candidate_node)
        if len(self.graph) > 0:
            max_degree = max(d for n, d in self.graph.degree())
            centrality_score = degree / max_degree if max_degree > 0 else 0.0
        else:
            centrality_score = 0.0

        # 3. Relationship Score
        relationship_score = self.get_relationship_score(candidate_node, seed_set)

        # Combined Score (weighted)
        combined = (0.5 * connectivity_score) + (0.3 * centrality_score) + (0.2 * relationship_score)
        
        return {
            "connectivity": connectivity_score,
            "centrality": centrality_score,
            "relationship": relationship_score,
            "combined": combined
        }

    def get_context_window(self, node_uuid: str, before: int = 2, after: int = 2) -> Dict[str, List[str]]:
        """
        Get the sliding context window around a node using next_chunk/previous_chunk edges.
        
        This implements "Solution B" for context fragmentation:
        By traversing sequential edges, we can retrieve the chunks before and after
        the matched chunk, ensuring headers get their content and vice versa.
        
        Args:
            node_uuid: The central node to get context around
            before: Number of previous chunks to retrieve (default 2)
            after: Number of next chunks to retrieve (default 2)
            
        Returns:
            Dict with 'before' (list of UUIDs), 'current', and 'after' (list of UUIDs)
        """
        if not self.graph.has_node(node_uuid):
            return {"before": [], "current": node_uuid, "after": []}
        
        # Get chunks BEFORE (traverse previous_chunk edges backwards)
        before_chunks = []
        current = node_uuid
        for _ in range(before):
            # Look for incoming previous_chunk or outgoing next_chunk TO us
            prev_found = None
            
            # Check predecessors with next_chunk relation pointing to us
            for pred in self.graph.predecessors(current):
                edge_data = self.graph.get_edge_data(pred, current)
                if edge_data and edge_data.get('relation') == 'next_chunk':
                    prev_found = pred
                    break
            
            # Also check if we have outgoing previous_chunk
            if not prev_found:
                for succ in self.graph.successors(current):
                    edge_data = self.graph.get_edge_data(current, succ)
                    if edge_data and edge_data.get('relation') == 'previous_chunk':
                        prev_found = succ
                        break
            
            if prev_found:
                before_chunks.insert(0, prev_found)  # Insert at front to maintain order
                current = prev_found
            else:
                break
        
        # Get chunks AFTER (traverse next_chunk edges forwards)
        after_chunks = []
        current = node_uuid
        for _ in range(after):
            next_found = None
            
            # Check successors with next_chunk relation
            for succ in self.graph.successors(current):
                edge_data = self.graph.get_edge_data(current, succ)
                if edge_data and edge_data.get('relation') == 'next_chunk':
                    next_found = succ
                    break
            
            # Also check if someone points to us with previous_chunk
            if not next_found:
                for pred in self.graph.predecessors(current):
                    edge_data = self.graph.get_edge_data(pred, current)
                    if edge_data and edge_data.get('relation') == 'previous_chunk':
                        next_found = pred
                        break
            
            if next_found:
                after_chunks.append(next_found)
                current = next_found
            else:
                break
        
        return {
            "before": before_chunks,
            "current": node_uuid,
            "after": after_chunks
        }

    def get_full_context_window(self, node_uuid: str, before: int = 2, after: int = 2) -> List[str]:
        """
        Get ordered list of all UUIDs in the context window.
        
        Args:
            node_uuid: Central node
            before: Chunks before (default 2)
            after: Chunks after (default 2)
            
        Returns:
            Ordered list: [before..., current, after...]
        """
        window = self.get_context_window(node_uuid, before, after)
        return window["before"] + [window["current"]] + window["after"]

graph_manager = GraphManager()
