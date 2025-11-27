import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import logging
from typing import List
from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
            cls._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return cls._instance

    def warmup(self):
        """Warmup the model to prevent cold start latency."""
        logger.info("Warming up embedding model...")
        dummy_sentences = ["Hello world", "Graph database", "Vector search"]
        self._model.encode(dummy_sentences)
        logger.info("Embedding model warmup complete.")

    @lru_cache(maxsize=1024)
    def _encode_cached(self, text: str) -> tuple:
        """Cached encoding returning a tuple (hashable)."""
        vector = self._model.encode(text)
        # L2 Normalization
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return tuple(vector)

    def encode(self, text: str) -> np.ndarray:
        """Public encode method returning numpy array."""
        vector_tuple = self._encode_cached(text)
        return np.array(vector_tuple, dtype=np.float32)

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple texts in batches for efficiency.
        Returns a 2D numpy array of shape (len(texts), embedding_dim).
        All vectors are L2 normalized.
        """
        if not texts:
            return np.array([], dtype=np.float32)
        
        # Check cache for already encoded texts
        cached_indices = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            # Check if in cache (without adding)
            try:
                # Try to get from cache without computing
                cache_info = self._encode_cached.cache_info()
                vector_tuple = self._encode_cached(text)
                cached_indices.append((i, vector_tuple))
            except:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Encode uncached texts in batches
        all_vectors = [None] * len(texts)
        
        # Place cached vectors
        for idx, vec_tuple in cached_indices:
            all_vectors[idx] = np.array(vec_tuple, dtype=np.float32)
        
        # Encode uncached in batches
        if uncached_texts:
            logger.info(f"Batch encoding {len(uncached_texts)} texts...")
            vectors = self._model.encode(
                uncached_texts,
                batch_size=batch_size,
                show_progress_bar=len(uncached_texts) > 100,
                normalize_embeddings=True  # L2 normalize
            )
            
            for idx, vec in zip(uncached_indices, vectors):
                all_vectors[idx] = vec.astype(np.float32)
                # Add to cache
                self._encode_cached(texts[idx])
        
        return np.vstack(all_vectors)

embedding_service = EmbeddingService()
