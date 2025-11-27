from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict
import uuid
import logging
import os

from models.api_schemas import NodeCreate, NodeResponse, EdgeCreate, SearchQuery, SearchResponse, BenchmarkRequest
from core.lock import read_locked, write_locked
from core.embedding import embedding_service
from core.fusion import hybrid_fusion
from storage.sqlite_ops import sqlite_manager
from storage.vector_ops import vector_index
from storage.graph_ops import graph_manager
from api.benchmark import calculate_metrics

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/v1/nodes", response_model=NodeResponse)
@write_locked
def create_node(node: NodeCreate):
    try:
        # 1. Generate UUID
        node_uuid = str(uuid.uuid4())
        
        # 2. Encode text
        vector = embedding_service.encode(node.text)
        
        # 3. SQLite Transaction (Atomic ID)
        faiss_id = sqlite_manager.add_node(
            uuid=node_uuid,
            text=node.text,
            node_type=node.type,
            metadata=node.metadata
        )
        
        # 4. FAISS Insertion
        vector_index.add_vector(vector, faiss_id, node_uuid)
        
        # 5. NetworkX Insertion
        graph_manager.add_node(node_uuid)
        
        return NodeResponse(
            uuid=node_uuid,
            faiss_id=faiss_id,
            text=node.text,
            type=node.type,
            metadata=node.metadata
        )
    except Exception as e:
        logger.error(f"Error creating node: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/v1/nodes/{node_uuid}", response_model=NodeResponse)
@read_locked
def get_node(node_uuid: str):
    node_data = sqlite_manager.get_node(node_uuid)
    if not node_data:
        raise HTTPException(status_code=404, detail="Node not found")
    return NodeResponse(**node_data)

@router.delete("/v1/nodes/{node_uuid}")
@write_locked
def delete_node(node_uuid: str):
    try:
        # 1. Remove from SQLite
        faiss_id = sqlite_manager.delete_node(node_uuid)
        if faiss_id is None:
            raise HTTPException(status_code=404, detail="Node not found")
            
        # 2. Remove from FAISS
        vector_index.remove_vector(faiss_id)
        
        # 3. Remove from NetworkX
        graph_manager.remove_node(node_uuid)
        
        return {"status": "deleted", "uuid": node_uuid}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting node: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/edges")
@write_locked
def create_edge(edge: EdgeCreate):
    try:
        # 1. Add to SQLite
        sqlite_manager.add_edge(edge.source_id, edge.target_id, edge.relation, edge.weight)
        
        # 2. Add to NetworkX
        graph_manager.add_edge(edge.source_id, edge.target_id, edge.relation, edge.weight)
        
        return {"status": "created", "edge": edge.dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating edge: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/search/hybrid", response_model=SearchResponse)
@read_locked
def hybrid_search(query: SearchQuery):
    try:
        # 1. Encode query
        query_vector = embedding_service.encode(query.text)
        
        # 2. FAISS Search (Initial Seeds)
        # Get top-50 as seeds for graph expansion
        initial_k = min(50, query.top_k * 5)
        seed_uuids, seed_scores = vector_index.search(query_vector, k=initial_k)
        
        if not seed_uuids:
            return SearchResponse(results=[], count=0)

        # Build seed score map for weighted graph scoring
        seed_score_map = {uuid: score for uuid, score in zip(seed_uuids, seed_scores)}
        
        # 3. Graph Expansion - Discover related nodes
        # Expand to depth=2 to find nodes not in initial vector results
        expansion_depth = 2
        expanded_candidates = graph_manager.expand_from_seeds(seed_uuids[:20], depth=expansion_depth)
        
        logger.info(f"Expanded from {len(seed_uuids)} seeds to {len(expanded_candidates)} candidates")
        
        # 4. Compute vector scores for ALL expanded candidates
        # For nodes already in seed results, use cached scores
        # For discovered nodes, compute on-the-fly
        new_nodes = [n for n in expanded_candidates if n not in seed_score_map]
        if new_nodes:
            new_scores = vector_index.batch_compute_similarity(query_vector, new_nodes)
            seed_score_map.update(new_scores)
        
        # 5. Hydrate all candidates with metadata
        all_candidates = []
        for uuid_str in expanded_candidates:
            node_data = sqlite_manager.get_node(uuid_str)
            if node_data:
                all_candidates.append({
                    "id": uuid_str,
                    "score": seed_score_map.get(uuid_str, 0.0),
                    "text": node_data['text'],
                    "metadata": node_data['metadata']
                })
        
        # 6. Calculate Graph Scores (with relationship awareness)
        # Use top-10 seeds for connectivity calculation
        top_seeds = seed_uuids[:10]
        graph_scores = {}
        
        for candidate in all_candidates:
            uuid_str = candidate['id']
            graph_breakdown = graph_manager.calculate_expanded_graph_score(
                uuid_str, 
                top_seeds,
                seed_vector_scores=seed_score_map
            )
            graph_scores[uuid_str] = graph_breakdown['combined']
            
        # 7. Fusion
        fused_results = hybrid_fusion(
            all_candidates, 
            graph_scores, 
            alpha=query.alpha, 
            beta=query.beta
        )
        
        # 8. Top K
        final_results = fused_results[:query.top_k]
        
        return SearchResponse(results=final_results, count=len(final_results))
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/search/hybrid/legacy", response_model=SearchResponse)
@read_locked
def hybrid_search_legacy(query: SearchQuery):
    """
    Legacy hybrid search (re-ranking only, no graph expansion).
    Kept for comparison purposes.
    """
    try:
        # 1. Encode query
        query_vector = embedding_service.encode(query.text)
        
        # 2. FAISS Search (Over-fetch)
        candidates_k = query.top_k * 3
        candidate_uuids, candidate_scores = vector_index.search(query_vector, k=candidates_k)
        
        if not candidate_uuids:
            return SearchResponse(results=[], count=0)

        # 3. Extract Seed Set (Top 10 from vector search)
        seed_set = candidate_uuids[:10]
        
        # 4. Hydrate candidates with metadata for fusion
        vector_results = []
        for uuid_str, score in zip(candidate_uuids, candidate_scores):
            node_data = sqlite_manager.get_node(uuid_str)
            if node_data:
                vector_results.append({
                    "id": uuid_str,
                    "score": score,
                    "text": node_data['text'],
                    "metadata": node_data['metadata']
                })
        
        # 5. Calculate Graph Scores
        graph_scores = {}
        for res in vector_results:
            uuid_str = res['id']
            g_score = graph_manager.calculate_graph_score(uuid_str, seed_set)
            graph_scores[uuid_str] = g_score
            
        # 6. Fusion
        fused_results = hybrid_fusion(
            vector_results, 
            graph_scores, 
            alpha=query.alpha, 
            beta=query.beta
        )
        
        # 7. Top K
        final_results = fused_results[:query.top_k]
        
        return SearchResponse(results=final_results, count=len(final_results))
        
    except Exception as e:
        logger.error(f"Error in legacy hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/v1/search/graph")
@read_locked
def graph_search(start_id: str, depth: int = 2, max_nodes: int = 50):
    return graph_manager.get_bfs_subgraph(start_id, depth, max_nodes)

@router.post("/v1/benchmark")
@read_locked
def run_benchmark(req: BenchmarkRequest):
    # This is a simplified benchmark runner that re-uses the search logic
    # In a real scenario, we might want to isolate components more strictly
    
    try:
        # 1. Vector Only (alpha=1.0, beta=0.0)
        vector_query = SearchQuery(text=req.query, top_k=max(req.k_values), alpha=1.0, beta=0.0)
        vector_res = hybrid_search(vector_query)
        vector_ids = [r.uuid for r in vector_res.results]
        
        # 2. Hybrid (alpha=0.7, beta=0.3)
        hybrid_query = SearchQuery(text=req.query, top_k=max(req.k_values), alpha=0.7, beta=0.3)
        hybrid_res = hybrid_search(hybrid_query)
        hybrid_ids = [r.uuid for r in hybrid_res.results]
        
        # Calculate metrics
        metrics = {
            "vector_only": calculate_metrics(vector_ids, req.ground_truth_ids, req.k_values),
            "hybrid": calculate_metrics(hybrid_ids, req.ground_truth_ids, req.k_values)
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error in benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/admin/snapshot")
@write_locked
def create_snapshot():
    try:
        # Save FAISS and Graph
        # SQLite is already persistent
        vector_index.save()
        graph_manager.save()
        
        # In a real distributed system, we would copy these to a snapshot location
        # Here we just ensure they are flushed to disk
        
        return {"status": "snapshot_created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "nodes": sqlite_manager.count_nodes(),
        "edges": sqlite_manager.count_edges()
    }

@router.get("/v1/nodes", response_model=List[Dict])
@read_locked
def list_nodes(limit: int = 100, offset: int = 0):
    """List all nodes with pagination."""
    try:
        return sqlite_manager.get_all_nodes(limit, offset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/v1/edges", response_model=List[Dict])
@read_locked
def list_edges(limit: int = 100, offset: int = 0):
    """List all edges with pagination."""
    try:
        return sqlite_manager.get_all_edges(limit, offset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
