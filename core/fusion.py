import numpy as np
from typing import List, Dict, Any

def hybrid_fusion(
    vector_results: List[Dict[str, Any]], 
    graph_scores: Dict[str, float], 
    alpha: float = 0.7, 
    beta: float = 0.3
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
            "retrieval_rank": rank
        })

    # Sort by final score descending
    fused_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Update retrieval_rank after sorting
    for rank, result in enumerate(fused_results, 1):
        result['retrieval_rank'] = rank
    
    return fused_results
