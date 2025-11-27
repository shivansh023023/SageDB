"""
Search utilities for deduplication, caching, and query decomposition.
"""
import hashlib
import json
import logging
import time
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from collections import OrderedDict
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================
# DEDUPLICATION
# ============================================

def deduplicate_results(
    results: List[Dict[str, Any]], 
    threshold: float = 0.95,
    embedding_service = None
) -> List[Dict[str, Any]]:
    """
    Remove semantically duplicate results.
    
    Two results are considered duplicates if their text embeddings
    have cosine similarity >= threshold.
    
    Uses a greedy approach: keep first result, remove subsequent
    results that are too similar to any kept result.
    
    Args:
        results: List of search results with 'text' field
        threshold: Similarity threshold for deduplication (default 0.95)
        embedding_service: Embedding service for computing similarity
        
    Returns:
        Deduplicated list of results
    """
    if len(results) <= 1 or embedding_service is None:
        return results
    
    # Get embeddings for all results
    texts = [r.get('text', '') for r in results]
    embeddings = [embedding_service.encode(t) for t in texts]
    
    kept_results = []
    kept_embeddings = []
    
    for i, (result, embedding) in enumerate(zip(results, embeddings)):
        # Check similarity to all kept results
        is_duplicate = False
        
        for kept_emb in kept_embeddings:
            similarity = float(np.dot(embedding, kept_emb))
            if similarity >= threshold:
                is_duplicate = True
                logger.debug(f"Dedup: Result {i} is duplicate (sim={similarity:.3f})")
                break
        
        if not is_duplicate:
            kept_results.append(result)
            kept_embeddings.append(embedding)
    
    logger.info(f"Deduplication: {len(results)} -> {len(kept_results)} results (threshold={threshold})")
    return kept_results


def deduplicate_by_text_hash(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fast deduplication using text hash (exact match only).
    Use as a first pass before semantic deduplication.
    """
    seen_hashes = set()
    unique_results = []
    
    for result in results:
        text = result.get('text', '')
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_results.append(result)
    
    return unique_results


# ============================================
# QUERY DECOMPOSITION
# ============================================

def decompose_query(query: str) -> List[str]:
    """
    Decompose a complex query into simpler sub-queries.
    
    Handles patterns like:
    - "Compare X and Y" -> ["X", "Y"]
    - "What is X and how does it relate to Y" -> ["What is X", "how does it relate to Y"]
    - "X vs Y" -> ["X", "Y"]
    - Questions with "and", "or" conjunctions
    
    Args:
        query: The original query string
        
    Returns:
        List of sub-queries (may be just the original query if no decomposition)
    """
    sub_queries = []
    
    # Pattern 1: "Compare X and Y" or "Compare X with Y"
    compare_match = re.match(r'compare\s+(.+?)\s+(?:and|with|to|vs\.?)\s+(.+)', query, re.IGNORECASE)
    if compare_match:
        sub_queries.append(compare_match.group(1).strip())
        sub_queries.append(compare_match.group(2).strip())
        logger.info(f"Query decomposed (compare): {query} -> {sub_queries}")
        return sub_queries
    
    # Pattern 2: "X vs Y" or "X versus Y"
    vs_match = re.match(r'(.+?)\s+(?:vs\.?|versus)\s+(.+)', query, re.IGNORECASE)
    if vs_match:
        sub_queries.append(vs_match.group(1).strip())
        sub_queries.append(vs_match.group(2).strip())
        logger.info(f"Query decomposed (vs): {query} -> {sub_queries}")
        return sub_queries
    
    # Pattern 3: "What is X and what is Y" - split on "and what/how/why"
    and_question = re.split(r'\s+and\s+(?:what|how|why|when|where)\s+', query, flags=re.IGNORECASE)
    if len(and_question) > 1:
        # Reconstruct each sub-query
        first = and_question[0].strip()
        for part in and_question[1:]:
            sub_queries.append(first)
            first = part.strip()
        sub_queries.append(first)
        logger.info(f"Query decomposed (and-question): {query} -> {sub_queries}")
        return sub_queries
    
    # Pattern 4: "X. Y." - Multiple sentences
    sentences = re.split(r'(?<=[.!?])\s+', query)
    if len(sentences) > 1:
        # Only decompose if each sentence is substantial (>10 chars)
        substantial = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(substantial) > 1:
            logger.info(f"Query decomposed (sentences): {query} -> {substantial}")
            return substantial
    
    # No decomposition - return original
    return [query]


def merge_sub_query_results(
    all_results: List[List[Dict[str, Any]]],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Merge results from multiple sub-queries using Round-Robin + score fusion.
    
    Args:
        all_results: List of result lists, one per sub-query
        top_k: Number of final results to return
        
    Returns:
        Merged and ranked results
    """
    if not all_results:
        return []
    
    if len(all_results) == 1:
        return all_results[0][:top_k]
    
    # Track seen UUIDs to avoid duplicates
    seen_uuids = set()
    merged = []
    
    # Score map for fusion (uuid -> list of scores from different sub-queries)
    score_map = {}
    result_map = {}
    
    # Collect all scores
    for results in all_results:
        for result in results:
            uuid = result['uuid']
            score = result.get('score', 0.0)
            
            if uuid not in score_map:
                score_map[uuid] = []
                result_map[uuid] = result
            
            score_map[uuid].append(score)
    
    # Fusion: average score across sub-queries, with bonus for appearing in multiple
    fused_scores = []
    for uuid, scores in score_map.items():
        avg_score = sum(scores) / len(scores)
        coverage_bonus = len(scores) / len(all_results) * 0.1  # Up to 0.1 bonus
        fused_score = avg_score + coverage_bonus
        fused_scores.append((uuid, fused_score))
    
    # Sort by fused score
    fused_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k
    merged = []
    for uuid, fused_score in fused_scores[:top_k]:
        result = result_map[uuid].copy()
        result['score'] = fused_score
        result['multi_query_fusion'] = True
        merged.append(result)
    
    return merged


# ============================================
# CACHING
# ============================================

class SearchCache:
    """
    LRU cache for search results with TTL support.
    
    Caches:
    - Query embeddings (expensive to compute)
    - Search results (for repeated queries)
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time-to-live in seconds (default 5 minutes)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        
        # Stats
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, query: str, params: Dict[str, Any]) -> str:
        """Create cache key from query and parameters."""
        key_data = {
            'query': query,
            'params': sorted(params.items())
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get(self, query: str, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached results if available and not expired.
        
        Args:
            query: Search query text
            params: Search parameters (top_k, alpha, beta, etc.)
            
        Returns:
            Cached results or None if not found/expired
        """
        key = self._make_key(query, params)
        
        if key in self._cache:
            # Check TTL
            timestamp = self._timestamps.get(key, 0)
            if time.time() - timestamp < self.ttl_seconds:
                # Move to end (LRU update)
                self._cache.move_to_end(key)
                self.hits += 1
                logger.debug(f"Cache HIT for query: {query[:50]}...")
                return self._cache[key]
            else:
                # Expired - remove
                self._remove(key)
        
        self.misses += 1
        return None
    
    def set(self, query: str, params: Dict[str, Any], results: List[Dict[str, Any]]):
        """
        Cache search results.
        
        Args:
            query: Search query text
            params: Search parameters
            results: Search results to cache
        """
        key = self._make_key(query, params)
        
        # Remove oldest if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            self._remove(oldest_key)
        
        self._cache[key] = results
        self._timestamps[key] = time.time()
    
    def _remove(self, key: str):
        """Remove a key from cache."""
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]
    
    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        self._timestamps.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'ttl_seconds': self.ttl_seconds
        }


# ============================================
# PROVENANCE TRACKING
# ============================================

class ProvenanceTracker:
    """
    Track retrieval provenance for citations and analytics.
    
    Records which chunks were retrieved, when, and with what scores.
    Enables:
    - Citation generation for RAG
    - Analytics on popular content
    - Feedback loop for relevance tuning
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize tracker.
        
        Args:
            max_history: Maximum number of retrievals to track
        """
        self.max_history = max_history
        self._history: List[Dict[str, Any]] = []
    
    def record_retrieval(
        self, 
        query: str, 
        results: List[Dict[str, Any]],
        search_type: str = "hybrid"
    ) -> str:
        """
        Record a retrieval event.
        
        Args:
            query: The search query
            results: List of retrieved results
            search_type: Type of search (hybrid, vector, etc.)
            
        Returns:
            Retrieval ID for future reference
        """
        retrieval_id = hashlib.sha256(
            f"{query}{time.time()}".encode()
        ).hexdigest()[:12]
        
        record = {
            'retrieval_id': retrieval_id,
            'timestamp': datetime.utcnow().isoformat(),
            'query': query,
            'query_hash': hashlib.md5(query.encode()).hexdigest()[:8],
            'search_type': search_type,
            'result_count': len(results),
            'chunks': [
                {
                    'uuid': r.get('uuid'),
                    'score': r.get('score'),
                    'rank': i + 1,
                    'source_document': r.get('source_document'),
                    'chunk_index': r.get('chunk_index')
                }
                for i, r in enumerate(results)
            ]
        }
        
        # Add to history
        self._history.append(record)
        
        # Trim if needed
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]
        
        return retrieval_id
    
    def get_retrieval(self, retrieval_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific retrieval record by ID."""
        for record in self._history:
            if record['retrieval_id'] == retrieval_id:
                return record
        return None
    
    def get_chunk_usage(self, uuid: str) -> Dict[str, Any]:
        """
        Get usage statistics for a specific chunk.
        
        Returns:
            Dict with retrieval count, average rank, average score, etc.
        """
        usages = []
        for record in self._history:
            for chunk in record['chunks']:
                if chunk['uuid'] == uuid:
                    usages.append({
                        'rank': chunk['rank'],
                        'score': chunk['score'],
                        'query_hash': record['query_hash'],
                        'timestamp': record['timestamp']
                    })
        
        if not usages:
            return {'uuid': uuid, 'retrieval_count': 0}
        
        return {
            'uuid': uuid,
            'retrieval_count': len(usages),
            'avg_rank': sum(u['rank'] for u in usages) / len(usages),
            'avg_score': sum(u['score'] for u in usages) / len(usages),
            'first_retrieved': min(u['timestamp'] for u in usages),
            'last_retrieved': max(u['timestamp'] for u in usages)
        }
    
    def get_popular_chunks(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently retrieved chunks."""
        chunk_counts = {}
        
        for record in self._history:
            for chunk in record['chunks']:
                uuid = chunk['uuid']
                if uuid not in chunk_counts:
                    chunk_counts[uuid] = {'count': 0, 'scores': [], 'ranks': []}
                chunk_counts[uuid]['count'] += 1
                chunk_counts[uuid]['scores'].append(chunk['score'])
                chunk_counts[uuid]['ranks'].append(chunk['rank'])
        
        # Sort by count
        sorted_chunks = sorted(
            chunk_counts.items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )[:top_k]
        
        return [
            {
                'uuid': uuid,
                'retrieval_count': data['count'],
                'avg_score': sum(data['scores']) / len(data['scores']),
                'avg_rank': sum(data['ranks']) / len(data['ranks'])
            }
            for uuid, data in sorted_chunks
        ]
    
    def export_history(self) -> List[Dict[str, Any]]:
        """Export full retrieval history."""
        return self._history.copy()


# Global instances
search_cache = SearchCache(max_size=100, ttl_seconds=300)
provenance_tracker = ProvenanceTracker(max_history=1000)
