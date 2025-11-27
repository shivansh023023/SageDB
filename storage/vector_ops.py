import faiss
import numpy as np
import os
import logging
from typing import List, Tuple
from config import FAISS_INDEX_PATH, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

class VectorIndex:
    def __init__(self, dimension: int = EMBEDDING_DIMENSION, index_path: str = FAISS_INDEX_PATH):
        self.dimension = dimension
        self.index_path = index_path
        self.index = None
        self.id_map = {}  # faiss_id -> uuid (In-memory map)

    def load_or_create(self, initial_id_map: dict = None):
        """Load index from disk or create new one."""
        if initial_id_map is not None:
            self.id_map = initial_id_map

        if os.path.exists(self.index_path):
            logger.info(f"Loading FAISS index from {self.index_path}")
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        logger.info("Creating new FAISS index")
        # IndexFlatIP: Inner Product (Cosine Similarity for normalized vectors)
        quantizer = faiss.IndexFlatIP(self.dimension)
        # IndexIDMap: Enables add_with_ids and remove_ids
        self.index = faiss.IndexIDMap(quantizer)

    def add_vector(self, vector: np.ndarray, faiss_id: int, uuid: str):
        """Add vector with specific ID."""
        # FAISS expects a matrix of vectors, so we reshape
        vector_matrix = vector.reshape(1, -1).astype('float32')
        id_array = np.array([faiss_id], dtype=np.int64)
        
        self.index.add_with_ids(vector_matrix, id_array)
        self.id_map[faiss_id] = uuid

    def remove_vector(self, faiss_id: int):
        """Remove vector by ID."""
        id_array = np.array([faiss_id], dtype=np.int64)
        self.index.remove_ids(id_array)
        if faiss_id in self.id_map:
            del self.id_map[faiss_id]

    def search(self, query_vector: np.ndarray, k: int) -> Tuple[List[str], List[float]]:
        """
        Search for nearest neighbors.
        Returns: (List[UUID], List[Scores])
        """
        query_matrix = query_vector.reshape(1, -1).astype('float32')
        scores, ids = self.index.search(query_matrix, k)
        
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

    def save(self):
        """Save index to disk."""
        faiss.write_index(self.index, self.index_path)

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

vector_index = VectorIndex()
