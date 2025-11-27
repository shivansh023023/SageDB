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
