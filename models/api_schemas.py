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
    alpha: float = Field(0.7, ge=0.0, le=1.0)
    beta: float = Field(0.3, ge=0.0, le=1.0)

    @validator('beta')
    def validate_weights(cls, v, values):
        if 'alpha' in values and abs(values['alpha'] + v - 1.0) > 1e-5:
            raise ValueError('Alpha and Beta must sum to 1.0')
        return v

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

class BenchmarkRequest(BaseModel):
    query: str
    ground_truth_ids: List[str]
    k_values: List[int] = [5, 10]
