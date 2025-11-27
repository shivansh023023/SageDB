import networkx as nx
import os
import logging
import math
import time
from typing import List, Dict, Optional
from config import (
    GRAPH_PATH, 
    PAGERANK_DAMPING, 
    PAGERANK_MAX_ITER, 
    PAGERANK_TOL,
    CENTRALITY_CACHE_TTL
)

logger = logging.getLogger(__name__)


class GraphManager:
    """
    Graph manager with enhanced features for scalability:
    
    1. GraphML Persistence: Uses GraphML format instead of pickle for:
       - Human-readable XML format
       - Interoperability with other graph tools (Gephi, Neo4j, etc.)
       - Robustness against Python version/library changes
       - Schema validation support
       
    2. PageRank Centrality: Replaces simple Degree Centrality with PageRank for:
       - Better importance scoring based on link quality, not just quantity
       - Handles graph structure more intelligently (endorsement propagation)
       - Standard algorithm used by Google, well-understood semantics
       
    3. Cached Centrality: PageRank is expensive O(N²), so we cache it:
       - Configurable TTL (default 5 minutes)
       - Automatically invalidates on graph modifications
       - Falls back to degree centrality if cache invalid and graph is large
    """
    
    def __init__(self, graph_path: str = GRAPH_PATH):
        self.graph_path = graph_path
        self.graph = nx.DiGraph()
        
        # PageRank cache for performance
        self._pagerank_cache: Dict[str, float] = {}
        self._pagerank_cache_time: float = 0
        self._pagerank_cache_valid: bool = False
        self._cache_ttl = CENTRALITY_CACHE_TTL
        
        # Track if graph was modified since last save
        self._modified = False

    def load_or_create(self):
        """Load graph from GraphML file or create new one."""
        # Try loading GraphML first
        if os.path.exists(self.graph_path):
            logger.info(f"Loading graph from {self.graph_path}")
            try:
                self.graph = nx.read_graphml(self.graph_path)
                # Convert to DiGraph if loaded as generic Graph
                if not isinstance(self.graph, nx.DiGraph):
                    self.graph = nx.DiGraph(self.graph)
                logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
                self._invalidate_pagerank_cache()
            except Exception as e:
                logger.error(f"Failed to load GraphML: {e}")
                self._try_legacy_load()
        else:
            # Check for legacy pickle file
            legacy_path = self.graph_path.replace('.graphml', '.gpickle')
            if os.path.exists(legacy_path):
                logger.info(f"Found legacy pickle file, migrating to GraphML...")
                self._try_legacy_load(legacy_path)
                # Save as GraphML after migration
                self.save()
            else:
                logger.info("Creating new NetworkX graph")
                self.graph = nx.DiGraph()
    
    def _try_legacy_load(self, legacy_path: str = None):
        """Try to load from legacy pickle format for migration."""
        if legacy_path is None:
            legacy_path = self.graph_path.replace('.graphml', '.gpickle')
        
        if os.path.exists(legacy_path):
            try:
                self.graph = nx.read_gpickle(legacy_path)
                logger.info(f"Migrated from legacy pickle: {self.graph.number_of_nodes()} nodes")
                self._invalidate_pagerank_cache()
            except Exception as e:
                logger.error(f"Failed to load legacy pickle: {e}")
                self.graph = nx.DiGraph()
        else:
            self.graph = nx.DiGraph()

    def add_node(self, uuid: str):
        self.graph.add_node(uuid)
        self._modified = True
        self._invalidate_pagerank_cache()

    def remove_node(self, uuid: str):
        if self.graph.has_node(uuid):
            self.graph.remove_node(uuid)
            self._modified = True
            self._invalidate_pagerank_cache()

    def add_edge(self, source: str, target: str, relation: str, weight: float):
        self.graph.add_edge(source, target, relation=relation, weight=weight)
        self._modified = True
        self._invalidate_pagerank_cache()

    def remove_edge(self, source: str, target: str):
        """Remove edge between two nodes if it exists."""
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
            self._modified = True
            self._invalidate_pagerank_cache()

    def save(self):
        """Save graph to GraphML format."""
        try:
            # Ensure all edge/node attributes are serializable
            # GraphML requires specific types (string, int, float, bool)
            self._prepare_for_graphml()
            nx.write_graphml(self.graph, self.graph_path)
            self._modified = False
            logger.info(f"Saved graph to {self.graph_path}")
        except Exception as e:
            logger.error(f"Failed to save GraphML: {e}")
            # Fallback to pickle if GraphML fails
            fallback_path = self.graph_path.replace('.graphml', '.gpickle')
            logger.info(f"Falling back to pickle: {fallback_path}")
            nx.write_gpickle(self.graph, fallback_path)
    
    def _prepare_for_graphml(self):
        """Prepare graph for GraphML serialization by ensuring proper types."""
        # GraphML supports: string, int, long, float, double, boolean
        for u, v, data in self.graph.edges(data=True):
            for key, value in list(data.items()):
                if value is None:
                    data[key] = ""
                elif not isinstance(value, (str, int, float, bool)):
                    data[key] = str(value)
        
        for node, data in self.graph.nodes(data=True):
            for key, value in list(data.items()):
                if value is None:
                    data[key] = ""
                elif not isinstance(value, (str, int, float, bool)):
                    data[key] = str(value)
    
    # ========================================
    # PAGERANK CENTRALITY (replaces Degree)
    # ========================================
    
    def _invalidate_pagerank_cache(self):
        """Mark PageRank cache as invalid."""
        self._pagerank_cache_valid = False
    
    def _compute_pagerank(self) -> Dict[str, float]:
        """
        Compute PageRank for all nodes.
        
        PageRank measures node importance based on:
        - Number of incoming edges (citations/references)
        - Quality of incoming edges (from high-PageRank nodes)
        
        This is superior to simple degree centrality because:
        - A link from an important node counts more
        - Handles "endorsement" semantics naturally
        - More robust to spam nodes with many low-quality links
        """
        if self.graph.number_of_nodes() == 0:
            return {}
        
        try:
            # Use personalized PageRank with uniform personalization
            pagerank = nx.pagerank(
                self.graph,
                alpha=PAGERANK_DAMPING,
                max_iter=PAGERANK_MAX_ITER,
                tol=PAGERANK_TOL,
                weight='weight'  # Use edge weights
            )
            return pagerank
        except Exception as e:
            logger.warning(f"PageRank computation failed: {e}, falling back to degree centrality")
            return self._compute_degree_centrality()
    
    def _compute_degree_centrality(self) -> Dict[str, float]:
        """Fallback: simple degree centrality."""
        if self.graph.number_of_nodes() == 0:
            return {}
        
        max_degree = max((d for n, d in self.graph.degree()), default=1)
        return {
            node: (self.graph.in_degree(node) + self.graph.out_degree(node)) / max(max_degree, 1)
            for node in self.graph.nodes()
        }
    
    def get_pagerank_scores(self) -> Dict[str, float]:
        """
        Get PageRank scores with caching.
        
        Returns cached scores if:
        - Cache is valid
        - Cache is not expired (TTL)
        
        Otherwise recomputes PageRank.
        """
        current_time = time.time()
        
        # Check if cache is valid and not expired
        if (self._pagerank_cache_valid and 
            self._pagerank_cache and
            (current_time - self._pagerank_cache_time) < self._cache_ttl):
            return self._pagerank_cache
        
        # Recompute PageRank
        logger.debug("Recomputing PageRank centrality...")
        start = time.time()
        
        self._pagerank_cache = self._compute_pagerank()
        self._pagerank_cache_time = current_time
        self._pagerank_cache_valid = True
        
        elapsed = time.time() - start
        logger.debug(f"PageRank computed in {elapsed:.3f}s for {self.graph.number_of_nodes()} nodes")
        
        return self._pagerank_cache
    
    def get_centrality_score(self, node: str) -> float:
        """Get PageRank-based centrality score for a single node."""
        if not self.graph.has_node(node):
            return 0.0
        
        pagerank = self.get_pagerank_scores()
        
        # Normalize to 0-1 range (PageRank values can be very small)
        if not pagerank:
            return 0.0
        
        max_pr = max(pagerank.values())
        if max_pr == 0:
            return 0.0
        
        return pagerank.get(node, 0.0) / max_pr

    def calculate_graph_score(self, candidate_node: str, seed_set: List[str]) -> float:
        """
        Calculate graph score based on connectivity to seeds and PageRank centrality.
        Formula: 0.7 * connectivity + 0.3 * centrality
        
        Uses PageRank instead of simple degree centrality for better importance scoring.
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

        # 2. PageRank Centrality Score (replaces simple degree centrality)
        centrality_score = self.get_centrality_score(candidate_node)

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

        # 2. PageRank Centrality Score (using cached PageRank)
        centrality_score = self.get_centrality_score(candidate_node)

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
