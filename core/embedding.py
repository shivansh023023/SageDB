import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import logging
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

embedding_service = EmbeddingService()
