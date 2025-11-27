# Ingestion Orchestrator - Coordinates the full ingestion pipeline

import hashlib
import logging
import os
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .config import (
    IngestionConfig, SUPPORTED_EXTENSIONS, EDGE_WEIGHTS,
    MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB
)
from .parsers import get_parser, ContentSection
from .chunker import SemanticChunker, Chunk

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""
    success: bool
    document_id: Optional[int] = None
    filename: str = ""
    file_type: str = ""
    file_size: int = 0
    chunks_created: int = 0
    nodes_created: int = 0
    edges_created: int = 0
    processing_time_ms: float = 0
    message: str = ""
    chunk_uuids: List[str] = field(default_factory=list)
    
    @property
    def error(self) -> Optional[str]:
        """Alias for backwards compatibility."""
        return self.message if not self.success else None


class IngestionOrchestrator:
    """
    Orchestrates the full ingestion pipeline:
    1. Validate file
    2. Parse content
    3. Chunk sections
    4. Generate embeddings (batch)
    5. Store in SQLite, FAISS, NetworkX
    6. Create relationships
    """
    
    def __init__(self):
        """Initialize orchestrator with lazy-loaded dependencies."""
        self._sqlite_manager = None
        self._vector_index = None
        self._graph_manager = None
        self._embedding_service = None
        self.chunker = SemanticChunker()
    
    @property
    def sqlite_manager(self):
        if self._sqlite_manager is None:
            from storage.sqlite_ops import sqlite_manager
            self._sqlite_manager = sqlite_manager
        return self._sqlite_manager
    
    @property
    def vector_index(self):
        if self._vector_index is None:
            from storage.vector_ops import vector_index
            self._vector_index = vector_index
        return self._vector_index
    
    @property
    def graph_manager(self):
        if self._graph_manager is None:
            from storage.graph_ops import graph_manager
            self._graph_manager = graph_manager
        return self._graph_manager
    
    @property
    def embedding_service(self):
        if self._embedding_service is None:
            from core.embedding import embedding_service
            self._embedding_service = embedding_service
        return self._embedding_service
    
    def ingest_file(self, filename: str, content: bytes,
                    config: Optional[IngestionConfig] = None) -> IngestionResult:
        """
        Ingest a file into the database.
        
        Args:
            filename: Original filename
            content: Raw file content as bytes
            config: Optional ingestion configuration
            
        Returns:
            IngestionResult with details of the operation
        """
        start_time = time.time()
        config = config or IngestionConfig()
        
        try:
            config.validate()
        except ValueError as e:
            return IngestionResult(success=False, message=str(e), filename=filename)
        
        # 1. Validate file
        validation_result = self._validate_file(content, filename)
        if not validation_result[0]:
            return IngestionResult(
                success=False, 
                message=validation_result[1], 
                filename=filename
            )
        
        file_type, text_content = validation_result[1], validation_result[2]
        
        # 2. Check for duplicates
        content_hash = self._compute_hash(content)
        existing_doc = self.sqlite_manager.get_document_by_hash(content_hash)
        if existing_doc:
            return IngestionResult(
                success=False,
                message=f"File already ingested as document ID {existing_doc['id']}",
                filename=filename,
                document_id=existing_doc['id']
            )
        
        # 3. Parse content
        try:
            parser = get_parser(file_type)
            sections = parser.parse(text_content, filename)
        except Exception as e:
            logger.error(f"Parse error for {filename}: {e}")
            return IngestionResult(
                success=False,
                message=f"Failed to parse file: {str(e)}",
                filename=filename
            )
        
        if not sections:
            return IngestionResult(
                success=False,
                message="No content could be extracted from file",
                filename=filename
            )
        
        # 4. Chunk sections
        self.chunker = SemanticChunker(
            max_tokens=config.chunk_size,
            overlap_tokens=config.chunk_overlap
        )
        
        all_chunks: List[Chunk] = []
        for section in sections:
            section_chunks = self.chunker.chunk_text(
                text=section.text,
                title=section.title,
                hierarchy=section.hierarchy,
                metadata=section.metadata
            )
            all_chunks.extend(section_chunks)
        
        if not all_chunks:
            return IngestionResult(
                success=False,
                message="No chunks could be created from content",
                filename=filename
            )
        
        # 5. Generate embeddings (batch)
        try:
            texts = [chunk.text for chunk in all_chunks]
            embeddings = self.embedding_service.encode_batch(texts)
        except Exception as e:
            logger.error(f"Embedding error for {filename}: {e}")
            return IngestionResult(
                success=False,
                message=f"Failed to generate embeddings: {str(e)}",
                filename=filename
            )
        
        # 6. Create document record
        try:
            document_id = self.sqlite_manager.add_document(
                filename=filename,
                file_type=file_type,
                content_hash=content_hash,
                original_size=len(content),
                metadata={"chunk_count": len(all_chunks)}
            )
        except Exception as e:
            logger.error(f"Failed to create document record: {e}")
            return IngestionResult(
                success=False,
                message=f"Failed to create document record: {str(e)}",
                filename=filename
            )
        
        # 7. Store chunks as nodes
        chunk_uuids = []
        nodes_created = 0
        
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            chunk_uuid = str(uuid.uuid4())
            
            try:
                # Prepare metadata
                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "title": chunk.title,
                    "hierarchy": chunk.hierarchy,
                    **(chunk.metadata or {})
                }
                
                # Add node to SQLite
                faiss_id = self.sqlite_manager.add_node(
                    uuid=chunk_uuid,
                    text=chunk.text,
                    node_type="document",
                    metadata=chunk_metadata
                )
                
                # Add chunk record (links node to document)
                self.sqlite_manager.add_chunk(
                    document_id=document_id,
                    node_uuid=chunk_uuid,
                    chunk_index=i,
                    section_hierarchy=">".join(chunk.hierarchy) if chunk.hierarchy else None
                )
                
                # Add to FAISS
                self.vector_index.add_vector(embedding, faiss_id, chunk_uuid)
                
                # Add to NetworkX
                self.graph_manager.add_node(chunk_uuid)
                
                chunk_uuids.append(chunk_uuid)
                nodes_created += 1
                
            except Exception as e:
                logger.error(f"Failed to store chunk {i}: {e}")
                # Continue with other chunks
        
        # 8. Create relationships
        edges_created = 0
        if config.auto_create_relationships:
            edges_created = self._create_relationships(
                document_id, chunk_uuids, all_chunks, config
            )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        logger.info(f"Ingested {filename}: {nodes_created} nodes, {edges_created} edges in {processing_time:.0f}ms")
        
        return IngestionResult(
            success=True,
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            file_size=len(content),
            chunks_created=len(chunk_uuids),
            nodes_created=nodes_created,
            edges_created=edges_created,
            processing_time_ms=processing_time,
            message=f"Successfully ingested {filename}",
            chunk_uuids=chunk_uuids
        )
    
    def _validate_file(self, content: bytes, filename: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate file before processing.
        
        Returns:
            Tuple of (is_valid, file_type_or_error, text_content)
        """
        # Check file size
        if len(content) > MAX_FILE_SIZE_BYTES:
            return (False, f"File exceeds maximum size of {MAX_FILE_SIZE_MB}MB", None)
        
        # Get file extension
        _, ext = os.path.splitext(filename.lower())
        if ext not in SUPPORTED_EXTENSIONS:
            return (False, f"Unsupported file type: {ext}. Supported: {list(SUPPORTED_EXTENSIONS.keys())}", None)
        
        file_type = SUPPORTED_EXTENSIONS[ext]
        
        # Decode content
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text_content = content.decode('latin-1')
            except:
                return (False, "Could not decode file content. Ensure file is text-based.", None)
        
        if not text_content.strip():
            return (False, "File is empty", None)
        
        return (True, file_type, text_content)
    
    def _compute_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of content for deduplication."""
        return hashlib.sha256(content).hexdigest()
    
    def _create_relationships(self, document_id: int, 
                             chunk_uuids: List[str], 
                             chunks: List[Chunk],
                             config: IngestionConfig) -> int:
        """
        Create relationships between chunks.
        
        Returns:
            Number of edges created
        """
        edges_created = 0
        
        # Create sequential relationships (next_chunk)
        if config.create_sequential_edges:
            for i in range(len(chunk_uuids) - 1):
                try:
                    self.sqlite_manager.add_edge(
                        source_uuid=chunk_uuids[i],
                        target_uuid=chunk_uuids[i + 1],
                        relation="next_chunk",
                        weight=EDGE_WEIGHTS["next_chunk"]
                    )
                    self.graph_manager.add_edge(
                        chunk_uuids[i], chunk_uuids[i + 1],
                        relation="next_chunk",
                        weight=EDGE_WEIGHTS["next_chunk"]
                    )
                    edges_created += 1
                except Exception as e:
                    logger.warning(f"Failed to create next_chunk edge: {e}")
        
        # Create section_of relationships based on hierarchy
        if config.preserve_hierarchy:
            hierarchy_map: Dict[str, str] = {}  # title -> uuid
            
            for chunk_uuid, chunk in zip(chunk_uuids, chunks):
                if chunk.hierarchy and len(chunk.hierarchy) > 1:
                    # This chunk's parent is the previous level in hierarchy
                    parent_title = chunk.hierarchy[-2] if len(chunk.hierarchy) >= 2 else None
                    
                    if parent_title and parent_title in hierarchy_map:
                        try:
                            parent_uuid = hierarchy_map[parent_title]
                            self.sqlite_manager.add_edge(
                                source_uuid=chunk_uuid,
                                target_uuid=parent_uuid,
                                relation="section_of",
                                weight=EDGE_WEIGHTS["section_of"]
                            )
                            self.graph_manager.add_edge(
                                chunk_uuid, parent_uuid,
                                relation="section_of",
                                weight=EDGE_WEIGHTS["section_of"]
                            )
                            edges_created += 1
                        except Exception as e:
                            logger.warning(f"Failed to create section_of edge: {e}")
                
                # Track this chunk's title for future parent lookups
                if chunk.title:
                    hierarchy_map[chunk.title] = chunk_uuid
        
        return edges_created
    
    def ingest_text(self, text: str, source_name: str = "user_input",
                    config: Optional[IngestionConfig] = None) -> IngestionResult:
        """
        Ingest raw text directly (without file upload).
        
        Args:
            text: Raw text content
            source_name: Name to identify the source
            config: Optional ingestion configuration
            
        Returns:
            IngestionResult with details of the operation
        """
        # Convert to bytes and treat as plain text file
        content = text.encode('utf-8')
        filename = f"{source_name}.txt"
        return self.ingest_file(filename, content, config)
