# Ingestion Pipeline Configuration

from dataclasses import dataclass
from typing import Dict

# File size limits
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Chunking defaults
DEFAULT_CHUNK_SIZE = 512  # tokens
DEFAULT_CHUNK_OVERLAP = 50  # tokens
MIN_CHUNK_SIZE = 50  # Don't create tiny chunks

# Supported file types
SUPPORTED_EXTENSIONS = {
    '.md': 'markdown',
    '.markdown': 'markdown',
    '.txt': 'text',
    '.text': 'text',
    '.html': 'html',
    '.htm': 'html',
    '.json': 'json',
    '.xml': 'xml',
}

# Edge weights for auto-created relationships
# Optimized for hybrid semantic + graph search
EDGE_WEIGHTS: Dict[str, float] = {
    # Parent-child (document structure) - HIGHEST
    "chunk_of": 1.0,          # Chunk belongs to document
    "section_of": 0.95,       # Section within a larger section (reduced to prioritize semantic)
    
    # Semantic similarity (auto-detected) - VERY HIGH
    "similar_to": 0.90,       # Strong semantic overlap (cosine > 0.8)
    "related_to": 0.75,       # Moderate semantic similarity (cosine > 0.75)
    
    # Sequential (reading order) - HIGH
    "next_chunk": 0.85,       # Sequential reading flow (boosted by similarity in code)
    "previous_chunk": 0.85,   # Backward reference
    
    # Taxonomic (knowledge structure)
    "is_a": 0.93,             # Classification hierarchy
    "part_of": 0.90,          # Compositional
    "instance_of": 0.88,      # Specific instance of concept
    
    # Functional
    "uses": 0.80,
    "depends_on": 0.80,
    "implements": 0.75,
    "extends": 0.70,
    
    # Reference
    "references": 0.65,       # Explicit link/citation
    "mentions": 0.40,         # Weak reference
}

# Minimum relevance threshold for search results
# Results below this raw similarity score are filtered out
MINIMUM_RELEVANCE_THRESHOLD = 0.25


@dataclass
class IngestionConfig:
    """Configuration for a single ingestion operation."""
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    auto_create_relationships: bool = True
    preserve_hierarchy: bool = True
    create_sequential_edges: bool = True  # Create next_chunk edges between consecutive chunks
    
    def validate(self):
        """Validate configuration values."""
        if self.chunk_size < MIN_CHUNK_SIZE:
            raise ValueError(f"chunk_size must be at least {MIN_CHUNK_SIZE}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
