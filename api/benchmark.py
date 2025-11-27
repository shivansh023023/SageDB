from typing import List, Dict
import math

def calculate_metrics(retrieved_ids: List[str], ground_truth_ids: List[str], k_values: List[int]) -> Dict:
    metrics = {}
    gt_set = set(ground_truth_ids)
    
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
        
        metrics[f"precision@{k}"] = precision
        metrics[f"recall@{k}"] = recall
        metrics[f"ndcg@{k}"] = ndcg
        
    return metrics
