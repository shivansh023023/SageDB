import faiss
import numpy as np
import os
import json
import logging
from typing import List, Tuple
from config import (
    FAISS_INDEX_PATH, 
    EMBEDDING_DIMENSION,
    FAISS_INDEX_TYPE,
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH
)

logger = logging.getLogger(__name__)

class VectorIndex:
    """
    Vector index manager supporting both brute-force (Flat) and approximate (HNSW) search.
    
    HNSW (Hierarchical Navigable Small World) provides O(log N) search complexity
    compared to O(N) for brute-force, enabling scalability to millions of vectors.
    
    Index Types:
    - "flat": IndexFlatIP wrapped in IndexIDMap (100% recall, O(N) search)
    - "hnsw": IndexHNSWFlat wrapped in IndexIDMap2 (approximate, O(log N) search)
    
    HNSW Parameters (tunable via config):
    - M: Number of bi-directional links per node (default 32)
    - efConstruction: Construction-time search depth (default 200)
    - efSearch: Query-time search depth (default 64, can be adjusted per-query)
    """
    
    def __init__(
        self, 
        dimension: int = EMBEDDING_DIMENSION, 
        index_path: str = FAISS_INDEX_PATH,
        index_type: str = FAISS_INDEX_TYPE
    ):
        self.dimension = dimension
        self.index_path = index_path
        self.index_type = index_type.lower()
        self.index = None
        self.id_map = {}  # faiss_id -> uuid (In-memory map)
        
        # HNSW parameters
        self.hnsw_m = HNSW_M
        self.hnsw_ef_construction = HNSW_EF_CONSTRUCTION
        self.hnsw_ef_search = HNSW_EF_SEARCH
        
        logger.info(f"VectorIndex initialized with type={self.index_type}, dimension={self.dimension}")

    def load_or_create(self, initial_id_map: dict = None):
        """Load index from disk or create new one."""
        if initial_id_map is not None:
            self.id_map = initial_id_map

        id_map_path = self.index_path + ".idmap"

        if os.path.exists(self.index_path):
            logger.info(f"Loading FAISS index from {self.index_path}")
            try:
                self.index = faiss.read_index(self.index_path)
                self._configure_search_params()
                
                # Load id_map if exists
                if os.path.exists(id_map_path):
                    with open(id_map_path, 'r') as f:
                        id_map_json = json.load(f)
                        # Convert str keys back to int
                        self.id_map = {int(k): v for k, v in id_map_json.items()}
                    logger.info(f"Loaded id_map with {len(self.id_map)} mappings")
                
                logger.info(f"Loaded index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index based on configured type."""
        if self.index_type == "hnsw":
            self._create_hnsw_index()
        else:
            self._create_flat_index()
    
    def _create_flat_index(self):
        """Create brute-force IndexFlatIP (100% recall, O(N) search)."""
        logger.info("Creating new FAISS IndexFlatIP (brute-force)")
        quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(quantizer)
    
    def _create_hnsw_index(self):
        """
        Create HNSW index for approximate nearest neighbor search.
        
        HNSW provides:
        - O(log N) search complexity (vs O(N) for flat)
        - Configurable accuracy/speed tradeoff via efSearch
        - Scales to millions of vectors
        
        Note: HNSW doesn't support remove_ids natively. We use IndexIDMap2
        which maintains a mapping and marks deleted IDs.
        """
        logger.info(
            f"Creating new FAISS IndexHNSWFlat (M={self.hnsw_m}, "
            f"efConstruction={self.hnsw_ef_construction})"
        )
        
        # Create HNSW index with Inner Product metric
        # Note: For cosine similarity with normalized vectors, IP = cosine similarity
        hnsw_index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        
        # Set construction-time parameters
        hnsw_index.hnsw.efConstruction = self.hnsw_ef_construction
        
        # Wrap in IndexIDMap2 to support custom IDs and deletion
        # IndexIDMap2 maintains a reverse mapping for ID operations
        self.index = faiss.IndexIDMap2(hnsw_index)
        
        self._configure_search_params()
    
    def _configure_search_params(self):
        """Configure search-time parameters for the index."""
        if self.index_type == "hnsw":
            try:
                # For IndexIDMap2 wrapping HNSW, access the underlying index
                underlying = faiss.downcast_index(self.index.index)
                if hasattr(underlying, 'hnsw'):
                    underlying.hnsw.efSearch = self.hnsw_ef_search
                    logger.info(f"Set HNSW efSearch={self.hnsw_ef_search}")
            except Exception as e:
                logger.warning(f"Could not set HNSW search params: {e}")

    def add_vector(self, vector: np.ndarray, faiss_id: int, uuid: str):
        """Add vector with specific ID."""
        # FAISS expects a matrix of vectors, so we reshape
        vector_matrix = vector.reshape(1, -1).astype('float32')
        id_array = np.array([faiss_id], dtype=np.int64)
        
        self.index.add_with_ids(vector_matrix, id_array)
        self.id_map[faiss_id] = uuid

    def remove_vector(self, faiss_id: int):
        """
        Remove vector by ID.
        
        Note: For HNSW indices wrapped in IndexIDMap2, this marks the ID as removed
        but doesn't reclaim memory until the index is rebuilt.
        """
        id_array = np.array([faiss_id], dtype=np.int64)
        try:
            self.index.remove_ids(id_array)
        except Exception as e:
            logger.warning(f"Could not remove vector {faiss_id}: {e}")
        
        if faiss_id in self.id_map:
            del self.id_map[faiss_id]

    def search(self, query_vector: np.ndarray, k: int, ef_search: int = None) -> Tuple[List[str], List[float]]:
        """
        Search for nearest neighbors.
        
        Args:
            query_vector: Query embedding
            k: Number of results to return
            ef_search: (HNSW only) Override efSearch for this query.
                       Higher values = more accurate but slower.
        
        Returns: (List[UUID], List[Scores])
        """
        # Temporarily adjust efSearch if specified (HNSW only)
        if ef_search is not None and self.index_type == "hnsw":
            self._set_ef_search(ef_search)
        
        query_matrix = query_vector.reshape(1, -1).astype('float32')
        scores, ids = self.index.search(query_matrix, k)
        
        # Restore default efSearch
        if ef_search is not None and self.index_type == "hnsw":
            self._set_ef_search(self.hnsw_ef_search)
        
        # Flatten results
        found_ids = ids[0]
        found_scores = scores[0]
        
        result_uuids = []
        result_scores = []
        
        for fid, score in zip(found_ids, found_scores):
            if fid != -1 and fid in self.id_map:
                result_uuids.append(self.id_map[fid])
                result_scores.append(float(score))
                
        return result_uuids, result_scores

    def search_with_filter(
        self, 
        query_vector: np.ndarray, 
        k: int, 
        allowed_faiss_ids: List[int],
        ef_search: int = None
    ) -> Tuple[List[str], List[float]]:
        """
        Search for nearest neighbors, but only among allowed FAISS IDs.
        
        This enables pre-filtered search where SQLite filters by metadata
        first, then we only search among the allowed vectors.
        
        Args:
            query_vector: Query embedding
            k: Number of results to return
            allowed_faiss_ids: List of FAISS IDs to consider (from SQLite pre-filter)
            ef_search: (HNSW only) Override efSearch for this query
            
        Returns: (List[UUID], List[Scores])
        """
        if not allowed_faiss_ids:
            return [], []
        
        # For small filter sets, compute scores directly
        if len(allowed_faiss_ids) <= 1000:
            return self._search_filtered_direct(query_vector, k, allowed_faiss_ids)
        
        # For larger sets, use IDSelectorBatch if available
        return self._search_filtered_batch(query_vector, k, allowed_faiss_ids, ef_search)
    
    def _search_filtered_direct(
        self, 
        query_vector: np.ndarray, 
        k: int, 
        allowed_faiss_ids: List[int]
    ) -> Tuple[List[str], List[float]]:
        """
        Direct filtered search for small filter sets.
        Computes similarity to each allowed vector and returns top-k.
        """
        query_norm = query_vector.reshape(-1).astype('float32')
        
        scored_results = []
        for faiss_id in allowed_faiss_ids:
            if faiss_id not in self.id_map:
                continue
            
            target_vector = self.get_vector_by_id(faiss_id)
            if target_vector is None:
                continue
            
            score = float(np.dot(query_norm, target_vector.reshape(-1)))
            scored_results.append((self.id_map[faiss_id], score))
        
        # Sort by score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        top_k = scored_results[:k]
        return [r[0] for r in top_k], [r[1] for r in top_k]
    
    def _search_filtered_batch(
        self, 
        query_vector: np.ndarray, 
        k: int, 
        allowed_faiss_ids: List[int],
        ef_search: int = None
    ) -> Tuple[List[str], List[float]]:
        """
        Batch filtered search for larger filter sets.
        Uses FAISS IDSelectorBatch for efficient filtering.
        """
        try:
            # Create ID selector for allowed IDs
            id_array = np.array(allowed_faiss_ids, dtype=np.int64)
            selector = faiss.IDSelectorBatch(id_array)
            
            # Create search parameters with selector
            params = faiss.SearchParametersIVF()
            params.sel = selector
            
            if ef_search is not None and self.index_type == "hnsw":
                self._set_ef_search(ef_search)
            
            query_matrix = query_vector.reshape(1, -1).astype('float32')
            
            # Search with selector
            scores, ids = self.index.search(query_matrix, k, params=params)
            
            if ef_search is not None and self.index_type == "hnsw":
                self._set_ef_search(self.hnsw_ef_search)
            
            result_uuids = []
            result_scores = []
            
            for fid, score in zip(ids[0], scores[0]):
                if fid != -1 and fid in self.id_map:
                    result_uuids.append(self.id_map[fid])
                    result_scores.append(float(score))
            
            return result_uuids, result_scores
            
        except Exception as e:
            # Fallback to direct method if IDSelectorBatch fails
            logger.warning(f"IDSelectorBatch failed: {e}, falling back to direct method")
            return self._search_filtered_direct(query_vector, k, allowed_faiss_ids)
    
    def _set_ef_search(self, ef_search: int):
        """Set efSearch parameter for HNSW index."""
        try:
            underlying = faiss.downcast_index(self.index.index)
            if hasattr(underlying, 'hnsw'):
                underlying.hnsw.efSearch = ef_search
        except Exception as e:
            pass  # Silently ignore for non-HNSW indices

    def save(self):
        """Save index and id_map to disk."""
        faiss.write_index(self.index, self.index_path)
        
        # Save id_map as JSON (convert int keys to str for JSON compatibility)
        id_map_path = self.index_path + ".idmap"
        id_map_json = {str(k): v for k, v in self.id_map.items()}
        with open(id_map_path, 'w') as f:
            json.dump(id_map_json, f)
        
        logger.info(f"Saved FAISS index to {self.index_path} ({self.index.ntotal} vectors, {len(self.id_map)} mappings)")

    def get_vector_by_id(self, faiss_id: int) -> np.ndarray:
        """
        Retrieve vector by FAISS ID.
        Note: This requires reconstructing from the index.
        """
        try:
            # For IndexIDMap, we need to reconstruct the vector
            vector = self.index.reconstruct(int(faiss_id))
            return vector
        except Exception as e:
            logger.warning(f"Could not reconstruct vector for ID {faiss_id}: {e}")
            return None

    def compute_similarity(self, query_vector: np.ndarray, target_uuid: str) -> float:
        """
        Compute cosine similarity between query vector and a specific node's vector.
        Returns similarity score (0-1 for normalized vectors).
        """
        # Find faiss_id from uuid
        target_faiss_id = None
        for fid, uuid in self.id_map.items():
            if uuid == target_uuid:
                target_faiss_id = fid
                break
        
        if target_faiss_id is None:
            return 0.0
        
        target_vector = self.get_vector_by_id(target_faiss_id)
        if target_vector is None:
            return 0.0
        
        # Compute inner product (cosine similarity for normalized vectors)
        query_norm = query_vector.reshape(-1).astype('float32')
        target_norm = target_vector.reshape(-1).astype('float32')
        similarity = float(np.dot(query_norm, target_norm))
        
        return max(0.0, similarity)  # Clamp to non-negative

    def batch_compute_similarity(self, query_vector: np.ndarray, target_uuids: List[str]) -> dict:
        """
        Compute similarity scores for multiple target nodes at once.
        Returns: Dict[uuid, similarity_score]
        """
        results = {}
        query_norm = query_vector.reshape(-1).astype('float32')
        
        for uuid in target_uuids:
            # Find faiss_id
            target_faiss_id = None
            for fid, u in self.id_map.items():
                if u == uuid:
                    target_faiss_id = fid
                    break
            
            if target_faiss_id is None:
                results[uuid] = 0.0
                continue
            
            target_vector = self.get_vector_by_id(target_faiss_id)
            if target_vector is None:
                results[uuid] = 0.0
                continue
            
            similarity = float(np.dot(query_norm, target_vector.reshape(-1)))
            results[uuid] = max(0.0, similarity)
        
        return results

    @property
    def ntotal(self):
        return self.index.ntotal
    
    def get_index_info(self) -> dict:
        """Get information about the current index for debugging/monitoring."""
        info = {
            "type": self.index_type,
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal,
            "id_map_size": len(self.id_map)
        }
        
        if self.index_type == "hnsw":
            info["hnsw_m"] = self.hnsw_m
            info["hnsw_ef_construction"] = self.hnsw_ef_construction
            info["hnsw_ef_search"] = self.hnsw_ef_search
            
        return info

vector_index = VectorIndex()
