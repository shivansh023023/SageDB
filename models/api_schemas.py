from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import re

class NodeCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    type: str = Field(..., pattern="^(document|entity|concept)$")
    metadata: Dict[str, str] = Field(default_factory=dict)

    @validator('text')
    def validate_text(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Text cannot be empty or whitespace only')
        return v

    @validator('metadata')
    def validate_metadata(cls, v):
        if len(v) > 10:
            raise ValueError('Metadata cannot have more than 10 keys')
        return v

class NodeResponse(BaseModel):
    uuid: str
    faiss_id: int
    text: str
    type: str
    metadata: Dict[str, str]

class EdgeCreate(BaseModel):
    source_id: str
    target_id: str
    relation: str = Field(..., min_length=1, max_length=50)
    weight: float = Field(..., gt=0.0, le=1.0)

    @validator('source_id', 'target_id')
    def validate_uuid(cls, v):
        uuid_regex = r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
        if not re.match(uuid_regex, v.lower()):
            raise ValueError('Invalid UUID v4 format')
        return v

class SearchQuery(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0, le=1000, description="Number of results to skip for pagination")
    alpha: float = Field(0.7, ge=0.0, le=1.0)
    beta: float = Field(0.3, ge=0.0, le=1.0)
    
    # Pre-filtering
    metadata_filter: Optional[Dict[str, str]] = Field(None, description="Filter by metadata before search (e.g., {'source': 'wiki'})")
    node_type_filter: Optional[str] = Field(None, pattern="^(document|entity|concept)$", description="Filter by node type")
    
    # Deduplication
    deduplicate: bool = Field(True, description="Remove semantically similar results (default: True)")
    dedup_threshold: float = Field(0.95, ge=0.5, le=1.0, description="Similarity threshold for deduplication")
    
    # Personalized PageRank
    use_ppr: bool = Field(True, description="Use Personalized PageRank for graph scoring (default: True)")
    
    # Query decomposition
    decompose_query: bool = Field(False, description="Decompose complex queries into sub-queries")
    
    # Caching
    bypass_cache: bool = Field(False, description="Skip cache lookup and force fresh search")

    # Note: alpha+beta no longer required to sum to 1.0
    # They are normalized internally to allow flexible weighting


class ContextSearchQuery(BaseModel):
    """Search query with context window expansion.
    
    Uses the "Sticky Header" solution to context fragmentation:
    For each result, retrieves surrounding chunks via graph traversal
    to ensure headers get their content and vice versa.
    """
    text: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0, le=1000, description="Number of results to skip for pagination")
    context_before: int = Field(2, ge=0, le=5, description="Chunks to include before each result")
    context_after: int = Field(2, ge=0, le=5, description="Chunks to include after each result")
    alpha: float = Field(0.7, ge=0.0, le=1.0)
    beta: float = Field(0.3, ge=0.0, le=1.0)


class VectorSearchQuery(BaseModel):
    """Vector-only search query."""
    text: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0, le=1000, description="Number of results to skip for pagination")


class NodeUpdate(BaseModel):
    """Update node metadata or text."""
    text: Optional[str] = Field(None, min_length=1, max_length=5000)
    metadata: Optional[Dict[str, str]] = None

    @validator('text')
    def validate_text(cls, v):
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError('Text cannot be empty or whitespace only')
        return v


class EdgeUpdate(BaseModel):
    """Update edge relation and/or weight."""
    relation: Optional[str] = Field(None, min_length=1, max_length=100)
    weight: Optional[float] = Field(None, gt=0.0, le=1.0)

    @validator('relation')
    def validate_relation(cls, v):
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError('Relation cannot be empty or whitespace only')
        return v


class EdgeResponse(BaseModel):
    """Edge response with ID."""
    id: int
    source_id: str
    target_id: str
    relation: str
    weight: float

class SearchResult(BaseModel):
    uuid: str
    text: str
    score: float
    vector_score: float
    graph_score: float
    metadata: Dict[str, Any]
    # Chunking strategy visibility fields
    source_document: Optional[str] = Field(None, description="Original document this chunk came from")
    chunk_index: Optional[int] = Field(None, description="Index of this chunk in the document")
    total_chunks: Optional[int] = Field(None, description="Total chunks in the source document")
    section_path: Optional[List[str]] = Field(None, description="Hierarchical section path (e.g., ['Chapter 1', 'Section 1.2'])")
    char_offset: Optional[int] = Field(None, description="Character offset in original document")
    # Provenance tracking
    retrieval_rank: Optional[int] = Field(None, description="Rank in which this result was retrieved")
    # Reasoning trace for explainability
    reasoning_trace: Optional[List[str]] = Field(None, description="Step-by-step reasoning path showing how this result was found")


class ReasoningStep(BaseModel):
    """A single step in the reasoning trace."""
    step_type: str = Field(..., description="Type: 'vector_match', 'graph_hop', 'ppr_boost', 'section_link'")
    description: str = Field(..., description="Human-readable explanation")
    score_contribution: Optional[float] = Field(None, description="Score contribution from this step")


class ChunkProvenance(BaseModel):
    """Provenance information for a retrieved chunk."""
    uuid: str
    score: float
    retrieval_rank: int
    timestamp: str
    query_hash: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    count: int


class ContextSearchResult(BaseModel):
    """Search result with context window expansion."""
    uuid: str
    text: str
    score: float
    vector_score: float
    graph_score: float
    metadata: Dict[str, Any]
    context_text: Optional[str] = Field(None, description="Combined text from context window")
    context_uuids: List[str] = Field(default_factory=list, description="UUIDs in context window")


class ContextSearchResponse(BaseModel):
    """Response with context-expanded results."""
    results: List[ContextSearchResult]
    count: int


class BenchmarkRequest(BaseModel):
    query: str
    ground_truth_ids: List[str]
    k_values: List[int] = [5, 10]
