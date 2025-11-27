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
    
    for res in vector_results:
        uuid = res['id']
        
        # Normalize vector score to [0, 1]
        v_score_norm = (res['score'] - min_v) / denom
        
        # Get graph score (already in [0, 1] or similar scale based on formula)
        g_score = graph_scores.get(uuid, 0.0)
        
        # Final Score Calculation
        final_score = (alpha * v_score_norm) + (beta * g_score)
        
        fused_results.append({
            "uuid": uuid,
            "text": res.get("text", ""),
            "metadata": res.get("metadata", {}),
            "score": final_score,
            "vector_score": v_score_norm,
            "graph_score": g_score,
            "raw_vector_score": res['score']
        })

    # Sort by final score descending
    fused_results.sort(key=lambda x: x['score'], reverse=True)
    
    return fused_results
