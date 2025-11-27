from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict
import uuid
import logging
import os

from models.api_schemas import (
    NodeCreate, NodeResponse, NodeUpdate, EdgeCreate, EdgeUpdate, EdgeResponse,
    SearchQuery, VectorSearchQuery, SearchResponse, BenchmarkRequest,
    ContextSearchQuery, ContextSearchResult, ContextSearchResponse
)
from core.lock import read_locked, write_locked
from core.embedding import embedding_service
from core.fusion import hybrid_fusion
from storage.sqlite_ops import sqlite_manager
from storage.vector_ops import vector_index
from storage.graph_ops import graph_manager
from api.benchmark import calculate_metrics
from ingestion.config import MINIMUM_RELEVANCE_THRESHOLD

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

@router.put("/v1/nodes/{node_uuid}")
@write_locked
def update_node(node_uuid: str, update: NodeUpdate):
    """Update node text and/or metadata. If text is updated, embedding is regenerated."""
    try:
        # Check if node exists
        existing = sqlite_manager.get_node(node_uuid)
        if not existing:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Update in SQLite
        updated = sqlite_manager.update_node(
            node_uuid, 
            text=update.text, 
            metadata=update.metadata
        )
        
        if not updated:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # If text changed, regenerate embedding
        if update.text is not None:
            new_vector = embedding_service.encode(update.text)
            vector_index.remove_vector(existing['faiss_id'])
            vector_index.add_vector(new_vector, existing['faiss_id'], node_uuid)
        
        # Return updated node
        updated_node = sqlite_manager.get_node(node_uuid)
        return NodeResponse(**updated_node)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating node: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

@router.post("/v1/edges", response_model=EdgeResponse)
@write_locked
def create_edge(edge: EdgeCreate):
    try:
        # 1. Add to SQLite (returns edge ID)
        edge_id = sqlite_manager.add_edge(edge.source_id, edge.target_id, edge.relation, edge.weight)
        
        # 2. Add to NetworkX
        graph_manager.add_edge(edge.source_id, edge.target_id, edge.relation, edge.weight)
        
        return EdgeResponse(
            id=edge_id,
            source_id=edge.source_id,
            target_id=edge.target_id,
            relation=edge.relation,
            weight=edge.weight
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating edge: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/v1/edges/{edge_id}", response_model=EdgeResponse)
@read_locked
def get_edge(edge_id: int):
    """Get edge by ID."""
    edge_data = sqlite_manager.get_edge(edge_id)
    if not edge_data:
        raise HTTPException(status_code=404, detail="Edge not found")
    return EdgeResponse(**edge_data)

@router.put("/v1/edges/{edge_id}", response_model=EdgeResponse)
@write_locked
def update_edge(edge_id: int, update: EdgeUpdate):
    """Update edge relation and/or weight."""
    try:
        if update.relation is None and update.weight is None:
            raise HTTPException(status_code=400, detail="At least one field (relation or weight) must be provided")
        
        # Get old edge data for NetworkX update
        old_edge = sqlite_manager.get_edge(edge_id)
        if not old_edge:
            raise HTTPException(status_code=404, detail="Edge not found")
        
        # Update in SQLite
        updated_edge = sqlite_manager.update_edge(edge_id, update.relation, update.weight)
        if not updated_edge:
            raise HTTPException(status_code=404, detail="Edge not found")
        
        # Update in NetworkX - remove old edge and add new one with updated attributes
        try:
            graph_manager.remove_edge(old_edge['source_id'], old_edge['target_id'])
            graph_manager.add_edge(
                updated_edge['source_id'],
                updated_edge['target_id'],
                updated_edge['relation'],
                updated_edge['weight']
            )
        except Exception as e:
            logger.warning(f"NetworkX edge update failed (continuing): {e}")
        
        return EdgeResponse(**updated_edge)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating edge: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/v1/edges/{edge_id}")
@write_locked
def delete_edge(edge_id: int):
    """Delete edge by ID."""
    try:
        # Get edge data first for NetworkX removal
        edge_data = sqlite_manager.get_edge(edge_id)
        if not edge_data:
            raise HTTPException(status_code=404, detail="Edge not found")
        
        # Remove from SQLite
        deleted = sqlite_manager.delete_edge(edge_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Edge not found")
        
        # Remove from NetworkX using proper encapsulation
        try:
            graph_manager.remove_edge(edge_data['source_id'], edge_data['target_id'])
        except Exception:
            pass  # Edge might not exist in graph
        
        return {"status": "deleted", "id": edge_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting edge: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/search/vector", response_model=SearchResponse)
@read_locked
def vector_search(query: VectorSearchQuery):
    """Vector-only search using cosine similarity with offset pagination."""
    try:
        # Encode query
        query_vector = embedding_service.encode(query.text)
        
        # FAISS Search - fetch extra to account for offset
        fetch_k = query.top_k + query.offset
        uuids, scores = vector_index.search(query_vector, k=fetch_k)
        
        if not uuids:
            return SearchResponse(results=[], count=0)
        
        # Filter by relevance threshold first
        filtered_pairs = [(u, s) for u, s in zip(uuids, scores) if s >= MINIMUM_RELEVANCE_THRESHOLD]
        if not filtered_pairs:
            return SearchResponse(results=[], count=0)
        
        # Apply offset pagination
        paginated_pairs = filtered_pairs[query.offset:query.offset + query.top_k]
        if not paginated_pairs:
            return SearchResponse(results=[], count=0)
        
        filtered_uuids = [u for u, s in paginated_pairs]
        
        # Batch hydrate with metadata (eliminates N+1 problem)
        node_data_map = sqlite_manager.get_nodes_batch(filtered_uuids)
        
        results = []
        for uuid_str, score in paginated_pairs:
            node_data = node_data_map.get(uuid_str)
            if node_data:
                results.append({
                    "uuid": uuid_str,
                    "text": node_data['text'],
                    "score": float(score),
                    "vector_score": float(score),
                    "graph_score": 0.0,  # Not used in vector-only search
                    "metadata": node_data['metadata']
                })
        
        return SearchResponse(results=results, count=len(results))
        
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/search/hybrid", response_model=SearchResponse)
@read_locked
def hybrid_search(query: SearchQuery):
    import time as _time
    timings = {}
    _t0 = _time.time()
    
    try:
        # Normalize alpha/beta if they don't sum to 1.0
        total = query.alpha + query.beta
        if total > 0:
            alpha = query.alpha / total
            beta = query.beta / total
        else:
            alpha, beta = 0.5, 0.5
        
        # 1. Encode query
        _t1 = _time.time()
        query_vector = embedding_service.encode(query.text)
        timings['1_encode'] = (_time.time() - _t1) * 1000
        
        # 2. FAISS Search (Initial Seeds)
        _t2 = _time.time()
        initial_k = min(50, query.top_k * 5)
        seed_uuids, seed_scores = vector_index.search(query_vector, k=initial_k)
        timings['2_faiss'] = (_time.time() - _t2) * 1000
        
        if not seed_uuids:
            return SearchResponse(results=[], count=0)

        # Build seed score map for weighted graph scoring
        seed_score_map = {uuid: score for uuid, score in zip(seed_uuids, seed_scores)}
        
        # 3. Graph Expansion
        _t3 = _time.time()
        expansion_depth = 2
        expanded_candidates = graph_manager.expand_from_seeds(seed_uuids[:20], depth=expansion_depth)
        timings['3_expand'] = (_time.time() - _t3) * 1000
        
        logger.info(f"Expanded from {len(seed_uuids)} seeds to {len(expanded_candidates)} candidates")
        
        # 4. Compute vector scores for expanded candidates
        _t4 = _time.time()
        new_nodes = [n for n in expanded_candidates if n not in seed_score_map]
        if new_nodes:
            new_scores = vector_index.batch_compute_similarity(query_vector, new_nodes)
            seed_score_map.update(new_scores)
        timings['4_batch_vec'] = (_time.time() - _t4) * 1000
        
        # 5. Batch Hydrate all candidates with metadata
        _t5 = _time.time()
        node_data_map = sqlite_manager.get_nodes_batch(list(expanded_candidates))
        timings['5_hydrate'] = (_time.time() - _t5) * 1000
        
        all_candidates = []
        for uuid_str in expanded_candidates:
            node_data = node_data_map.get(uuid_str)
            if node_data:
                all_candidates.append({
                    "id": uuid_str,
                    "score": seed_score_map.get(uuid_str, 0.0),
                    "text": node_data['text'],
                    "metadata": node_data['metadata']
                })
        
        # 6. Calculate Graph Scores (with relationship awareness)
        _t6 = _time.time()
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
        timings['6_graph_score'] = (_time.time() - _t6) * 1000
            
        # 7. Fusion (using normalized alpha/beta)
        _t7 = _time.time()
        fused_results = hybrid_fusion(
            all_candidates, 
            graph_scores, 
            alpha=alpha, 
            beta=beta
        )
        timings['7_fusion'] = (_time.time() - _t7) * 1000
        
        # 8. Filter out low relevance results
        filtered_results = [
            r for r in fused_results 
            if r.get('score', 0) >= MINIMUM_RELEVANCE_THRESHOLD
        ]
        
        # 9. Apply offset pagination and Top K
        final_results = filtered_results[query.offset:query.offset + query.top_k]
        
        timings['total'] = (_time.time() - _t0) * 1000
        logger.info(f"SEARCH TIMINGS: {timings}")
        
        return SearchResponse(results=final_results, count=len(final_results))
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/search/context", response_model=ContextSearchResponse)
@read_locked
def context_aware_search(query: ContextSearchQuery):
    """
    Context-aware hybrid search with sliding window expansion.
    
    This is the primary search endpoint for RAG applications.
    It implements "Solution B" for context fragmentation:
    
    For each matched chunk, we traverse next_chunk/previous_chunk edges
    to retrieve a context window (default: 2 before + current + 2 after = 5 chunks).
    
    This ensures that if a query matches a header, we also return the content,
    and if it matches content, we also return the header.
    """
    try:
        # Normalize alpha/beta
        total = query.alpha + query.beta
        if total > 0:
            alpha = query.alpha / total
            beta = query.beta / total
        else:
            alpha, beta = 0.5, 0.5
        
        # 1. Encode query
        query_vector = embedding_service.encode(query.text)
        
        # 2. FAISS Search (Initial Seeds)
        initial_k = min(50, query.top_k * 5)
        seed_uuids, seed_scores = vector_index.search(query_vector, k=initial_k)
        
        if not seed_uuids:
            return ContextSearchResponse(results=[], count=0)

        seed_score_map = {uuid: score for uuid, score in zip(seed_uuids, seed_scores)}
        
        # 3. Graph Expansion
        expansion_depth = 2
        expanded_candidates = graph_manager.expand_from_seeds(seed_uuids[:20], depth=expansion_depth)
        
        # 4. Compute vector scores for expanded candidates
        new_nodes = [n for n in expanded_candidates if n not in seed_score_map]
        if new_nodes:
            new_scores = vector_index.batch_compute_similarity(query_vector, new_nodes)
            seed_score_map.update(new_scores)
        
        # 5. Hydrate candidates
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
        
        # 6. Calculate Graph Scores
        top_seeds = seed_uuids[:10]
        graph_scores = {}
        for candidate in all_candidates:
            uuid_str = candidate['id']
            graph_breakdown = graph_manager.calculate_expanded_graph_score(
                uuid_str, top_seeds, seed_vector_scores=seed_score_map
            )
            graph_scores[uuid_str] = graph_breakdown['combined']
        
        # 7. Fusion
        fused_results = hybrid_fusion(
            all_candidates, graph_scores, alpha=alpha, beta=beta
        )
        
        # 8. Filter low relevance
        filtered_results = [
            r for r in fused_results 
            if r.get('score', 0) >= MINIMUM_RELEVANCE_THRESHOLD
        ]
        
        # 9. Apply offset pagination and take top K
        top_results = filtered_results[query.offset:query.offset + query.top_k]
        
        # 10. CONTEXT EXPANSION - The key feature!
        # For each result, get surrounding chunks via graph traversal
        
        # First, collect all context UUIDs we'll need
        all_context_uuids = set()
        result_context_windows = {}
        
        for result in top_results:
            uuid_str = result['uuid']
            context_window = graph_manager.get_full_context_window(
                uuid_str,
                before=query.context_before,
                after=query.context_after
            )
            result_context_windows[uuid_str] = context_window
            all_context_uuids.update(context_window)
        
        # Batch fetch all context nodes
        context_node_map = sqlite_manager.get_nodes_batch(list(all_context_uuids))
        
        # Now build results with context
        context_results = []
        seen_uuids = set()  # Avoid duplicating chunks across results
        
        for result in top_results:
            uuid_str = result['uuid']
            context_window = result_context_windows[uuid_str]
            
            # Build combined context text
            context_texts = []
            context_uuids = []
            
            for ctx_uuid in context_window:
                if ctx_uuid not in seen_uuids:
                    ctx_node = context_node_map.get(ctx_uuid)
                    if ctx_node:
                        context_texts.append(ctx_node['text'])
                        context_uuids.append(ctx_uuid)
                        seen_uuids.add(ctx_uuid)
            
            # Join context texts with separator
            context_text = "\n\n---\n\n".join(context_texts) if context_texts else result['text']
            
            context_results.append(ContextSearchResult(
                uuid=uuid_str,
                text=result['text'],
                score=result['score'],
                vector_score=result.get('vector_score', result['score']),
                graph_score=result.get('graph_score', 0.0),
                metadata=result['metadata'],
                context_text=context_text,
                context_uuids=context_uuids
            ))
        
        return ContextSearchResponse(results=context_results, count=len(context_results))
        
    except Exception as e:
        logger.error(f"Error in context-aware search: {e}")
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
        
        # 4. Batch hydrate candidates with metadata (eliminates N+1)
        node_data_map = sqlite_manager.get_nodes_batch(candidate_uuids)
        
        vector_results = []
        for uuid_str, score in zip(candidate_uuids, candidate_scores):
            node_data = node_data_map.get(uuid_str)
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
        
        # 7. Apply offset pagination and Top K
        final_results = fused_results[query.offset:query.offset + query.top_k]
        
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
