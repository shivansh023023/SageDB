from typing import List, Dict, Any, Optional
import math
import json
import os
import logging

logger = logging.getLogger(__name__)

# Path to ground truth file
GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ground_truth.json")


def load_ground_truth() -> Optional[Dict[str, Any]]:
    """Load the ground truth benchmark dataset."""
    try:
        if os.path.exists(GROUND_TRUTH_PATH):
            with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load ground truth: {e}")
    return None


def calculate_mrr(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank.
    MRR = 1/rank of first relevant result (0 if none found).
    """
    gt_set = set(ground_truth_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gt_set:
            return 1.0 / rank
    return 0.0


def calculate_metrics(retrieved_ids: List[str], ground_truth_ids: List[str], k_values: List[int]) -> Dict:
    """
    Calculate retrieval metrics for a single query.
    
    Args:
        retrieved_ids: List of retrieved document UUIDs
        ground_truth_ids: List of relevant document UUIDs
        k_values: List of K values for Precision@K, Recall@K, NDCG@K
        
    Returns:
        Dict with metrics for each K value plus MRR
    """
    metrics = {}
    gt_set = set(ground_truth_ids)
    
    # MRR (not K-dependent)
    metrics['mrr'] = calculate_mrr(retrieved_ids, ground_truth_ids)
    
    for k in k_values:
        # Slice top k
        top_k = retrieved_ids[:k]
        
        # Precision @ K
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in gt_set)
        precision = relevant_retrieved / k if k > 0 else 0.0
        
        # Recall @ K
        recall = relevant_retrieved / len(gt_set) if gt_set else 0.0
        
        # NDCG @ K
        dcg = 0.0
        idcg = 0.0
        
        for i, doc_id in enumerate(top_k):
            if doc_id in gt_set:
                dcg += 1.0 / math.log2(i + 2)
                
        for i in range(min(len(gt_set), k)):
            idcg += 1.0 / math.log2(i + 2)
            
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        metrics[f"precision@{k}"] = round(precision, 4)
        metrics[f"recall@{k}"] = round(recall, 4)
        metrics[f"ndcg@{k}"] = round(ndcg, 4)
        
    return metrics


def run_benchmark_comparison(
    search_func,
    queries: List[Dict[str, Any]],
    k_values: List[int] = [3, 5, 10]
) -> Dict[str, Any]:
    """
    Run a full benchmark comparing Vector-only vs Hybrid search.
    
    Args:
        search_func: Function that takes (query_text, search_type, alpha, beta, top_k) and returns result UUIDs
        queries: List of query dicts from ground_truth.json
        k_values: K values for metrics
        
    Returns:
        Dict with aggregated metrics for vector-only and hybrid modes
    """
    vector_metrics = {f"ndcg@{k}": [] for k in k_values}
    hybrid_metrics = {f"ndcg@{k}": [] for k in k_values}
    vector_mrrs = []
    hybrid_mrrs = []
    
    results_detail = []
    
    for q in queries:
        query_text = q['query']
        # Note: ground_truth_keywords are for display; actual ground_truth_ids would be UUIDs
        # For demo purposes, we'll use keyword matching to determine relevance
        expected_keywords = q.get('ground_truth_keywords', [])
        
        # This would need to be implemented with actual search
        # For now, we return a structure that the UI can display
        results_detail.append({
            'query_id': q['id'],
            'query': query_text,
            'scenario': q.get('scenario', 'unknown'),
            'expected_vector': q.get('expected_vector_performance', 'unknown'),
            'expected_hybrid': q.get('expected_hybrid_performance', 'unknown'),
            'keywords': expected_keywords
        })
    
    return {
        'queries': results_detail,
        'k_values': k_values,
        'summary': {
            'total_queries': len(queries),
            'scenarios': list(set(q.get('scenario', 'unknown') for q in queries))
        }
    }
