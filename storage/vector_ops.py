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

    @property
    def ntotal(self):
        return self.index.ntotal

vector_index = VectorIndex()
