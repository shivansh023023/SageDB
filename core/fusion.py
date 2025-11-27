import numpy as np
from typing import List, Dict, Any, Optional


def _generate_reasoning_trace(
    uuid: str,
    v_score_norm: float,
    g_score: float,
    final_score: float,
    metadata: Dict[str, Any],
    rank: int,
    use_ppr: bool = False,
    seed_info: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Generate a human-readable reasoning trace explaining why this result was retrieved.
    
    Args:
        uuid: Result UUID
        v_score_norm: Normalized vector score
        g_score: Graph score
        final_score: Final fused score
        metadata: Result metadata
        rank: Retrieval rank
        use_ppr: Whether Personalized PageRank was used
        seed_info: Optional info about PPR seed nodes
        
    Returns:
        List of reasoning steps as strings
    """
    trace = []
    
    # Step 1: Vector match explanation
    if v_score_norm >= 0.8:
        trace.append(f"🎯 Strong semantic match (vector score: {v_score_norm:.2f})")
    elif v_score_norm >= 0.5:
        trace.append(f"📊 Moderate semantic match (vector score: {v_score_norm:.2f})")
    else:
        trace.append(f"🔗 Weak semantic match (vector score: {v_score_norm:.2f}) - retrieved via graph")
    
    # Step 2: Graph contribution
    if g_score > 0:
        if use_ppr and g_score >= 0.2:
            trace.append(f"🌐 High graph centrality via Personalized PageRank (score: {g_score:.2f})")
        elif g_score >= 0.15:
            trace.append(f"🔀 Well-connected node in knowledge graph (PageRank: {g_score:.2f})")
        elif g_score > 0:
            trace.append(f"📎 Graph connection detected (score: {g_score:.2f})")
    
    # Step 3: Section/hierarchy context
    section_path = metadata.get("hierarchy") or metadata.get("section_path")
    if section_path:
        if isinstance(section_path, list):
            path_str = " → ".join(section_path)
        else:
            path_str = str(section_path)
        trace.append(f"📑 Found in section: {path_str}")
    
    # Step 4: Source context
    source = metadata.get("source")
    chunk_idx = metadata.get("chunk_index")
    if source:
        if chunk_idx is not None:
            trace.append(f"📄 Source: {source} (chunk {chunk_idx})")
        else:
            trace.append(f"📄 Source: {source}")
    
    # Step 5: Final score breakdown
    trace.append(f"⚡ Final score: {final_score:.3f} (rank #{rank})")
    
    return trace


def hybrid_fusion(
    vector_results: List[Dict[str, Any]], 
    graph_scores: Dict[str, float], 
    alpha: float = 0.7, 
    beta: float = 0.3,
    use_ppr: bool = False,
    seed_uuids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Combine vector similarity scores and graph scores using late interaction fusion.
    
    Args:
        vector_results: List of dicts with 'id', 'score' (raw vector score), and 'metadata'.
        graph_scores: Dict mapping UUID to calculated graph score.
        alpha: Weight for vector score.
        beta: Weight for graph score.
        
    Returns:
        List of results sorted by final score.
    """
    if not vector_results:
        return []

    # 1. Normalize Vector Scores (Min-Max Scaling)
    v_scores = [r['score'] for r in vector_results]
    min_v = min(v_scores) if v_scores else 0.0
    max_v = max(v_scores) if v_scores else 1.0
    
    # Avoid division by zero if all scores are identical
    denom = max_v - min_v if max_v != min_v else 1.0

    fused_results = []
    
    for rank, res in enumerate(vector_results, 1):
        uuid = res['id']
        raw_score = res['score']
        metadata = res.get("metadata", {})
        
        # Normalize vector score to [0, 1]
        v_score_norm = (raw_score - min_v) / denom
        
        # Get graph score (already in [0, 1] or similar scale based on formula)
        g_score = graph_scores.get(uuid, 0.0)
        
        # Final Score Calculation
        final_score = (alpha * v_score_norm) + (beta * g_score)
        
        # Extract chunking info from metadata
        source_document = metadata.get("source")
        chunk_index = metadata.get("chunk_index")
        total_chunks = metadata.get("total_chunks")
        char_offset = metadata.get("char_offset")
        
        # Generate reasoning trace for explainability
        reasoning_trace = _generate_reasoning_trace(
            uuid=uuid,
            v_score_norm=v_score_norm,
            g_score=g_score,
            final_score=final_score,
            metadata=metadata,
            rank=rank,
            use_ppr=use_ppr,
            seed_info={'seeds': seed_uuids} if seed_uuids else None
        )
        
        fused_results.append({
            "uuid": uuid,
            "text": res.get("text", ""),
            "metadata": metadata,
            "score": final_score,
            "vector_score": v_score_norm,
            "graph_score": g_score,
            "raw_vector_score": raw_score,
            # Chunking visibility fields
            "source_document": source_document,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "section_path": metadata.get("hierarchy"),
            "char_offset": char_offset,
            # Retrieval metadata
            "retrieval_rank": rank,
            # Explainability
            "reasoning_trace": reasoning_trace
        })

    # Sort by final score descending
    fused_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Update retrieval_rank and reasoning trace after sorting
    for rank, result in enumerate(fused_results, 1):
        result['retrieval_rank'] = rank
        # Update the final rank in reasoning trace
        if result.get('reasoning_trace'):
            result['reasoning_trace'][-1] = f"⚡ Final score: {result['score']:.3f} (rank #{rank})"
    
    return fused_results
