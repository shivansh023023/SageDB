"""
FastAPI router for ingestion endpoints.
Provides REST API for ingesting documents into SageDB.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import List, Optional
import logging

from .orchestrator import IngestionOrchestrator, IngestionResult
from .config import IngestionConfig, SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/ingest", tags=["ingestion"])

# Initialize orchestrator (lazy loaded on first request)
_orchestrator: Optional[IngestionOrchestrator] = None


def get_orchestrator() -> IngestionOrchestrator:
    """Get or create the ingestion orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = IngestionOrchestrator()
    return _orchestrator


class IngestionResponse(BaseModel):
    """Response model for ingestion results."""
    success: bool
    document_id: Optional[int] = None
    filename: str
    file_type: str
    chunks_created: int
    nodes_created: int
    edges_created: int
    message: str
    processing_time_ms: Optional[float] = None
    chunk_uuids: List[str] = []


class BatchIngestionResponse(BaseModel):
    """Response model for batch ingestion results."""
    total_files: int
    successful: int
    failed: int
    results: List[IngestionResponse]


class SupportedTypesResponse(BaseModel):
    """Response model for supported file types."""
    extensions: dict
    max_file_size_mb: int


@router.get("/supported-types", response_model=SupportedTypesResponse)
async def get_supported_types():
    """Get list of supported file types and limits."""
    return SupportedTypesResponse(
        extensions=SUPPORTED_EXTENSIONS,
        max_file_size_mb=MAX_FILE_SIZE_MB
    )


@router.post("/file", response_model=IngestionResponse)
async def ingest_file(
    file: UploadFile = File(...),
    create_sequential_edges: bool = Form(default=True)
):
    """
    Ingest a single file into SageDB.
    
    The file will be:
    1. Validated for size and type
    2. Parsed based on file extension
    3. Chunked into semantic segments
    4. Embedded and stored in vector store
    5. Connected with graph relationships
    
    Args:
        file: The file to ingest (supports .md, .txt, .html, .htm, .json, .xml)
        create_sequential_edges: Whether to create next_chunk edges between consecutive chunks
    
    Returns:
        IngestionResponse with details about the ingestion
    """
    import time
    start_time = time.time()
    
    orchestrator = get_orchestrator()
    
    try:
        # Read file content
        content = await file.read()
        
        # Create config
        config = IngestionConfig(create_sequential_edges=create_sequential_edges)
        
        # Ingest
        result = orchestrator.ingest_file(
            filename=file.filename,
            content=content,
            config=config
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return IngestionResponse(
            success=result.success,
            document_id=result.document_id,
            filename=result.filename,
            file_type=result.file_type,
            chunks_created=result.chunks_created,
            nodes_created=result.nodes_created,
            edges_created=result.edges_created,
            message=result.message,
            processing_time_ms=round(processing_time, 2),
            chunk_uuids=result.chunk_uuids
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error ingesting file {file.filename}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/batch", response_model=BatchIngestionResponse)
async def ingest_batch(
    files: List[UploadFile] = File(...),
    create_sequential_edges: bool = Form(default=True)
):
    """
    Ingest multiple files into SageDB.
    
    Each file will be processed independently. If one file fails,
    others will still be processed.
    
    Args:
        files: List of files to ingest
        create_sequential_edges: Whether to create next_chunk edges
    
    Returns:
        BatchIngestionResponse with results for each file
    """
    orchestrator = get_orchestrator()
    config = IngestionConfig(create_sequential_edges=create_sequential_edges)
    
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        import time
        start_time = time.time()
        
        try:
            content = await file.read()
            result = orchestrator.ingest_file(
                filename=file.filename,
                content=content,
                config=config
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            results.append(IngestionResponse(
                success=result.success,
                document_id=result.document_id,
                filename=result.filename,
                file_type=result.file_type,
                chunks_created=result.chunks_created,
                nodes_created=result.nodes_created,
                edges_created=result.edges_created,
                message=result.message,
                processing_time_ms=round(processing_time, 2),
                chunk_uuids=result.chunk_uuids
            ))
            
            if result.success:
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            logger.error(f"Error ingesting file {file.filename}: {e}")
            results.append(IngestionResponse(
                success=False,
                filename=file.filename,
                file_type="unknown",
                chunks_created=0,
                nodes_created=0,
                edges_created=0,
                message=str(e),
                chunk_uuids=[]
            ))
            failed += 1
    
    return BatchIngestionResponse(
        total_files=len(files),
        successful=successful,
        failed=failed,
        results=results
    )


@router.post("/text", response_model=IngestionResponse)
async def ingest_text(
    text: str = Form(...),
    filename: str = Form(default="input.txt"),
    file_type: str = Form(default="txt"),
    create_sequential_edges: bool = Form(default=True)
):
    """
    Ingest raw text directly into SageDB.
    
    Useful for ingesting text content without a file upload.
    
    Args:
        text: The text content to ingest
        filename: Optional filename to associate with the content
        file_type: The type of content (txt, md, html, json)
        create_sequential_edges: Whether to create next_chunk edges
    
    Returns:
        IngestionResponse with details about the ingestion
    """
    import time
    start_time = time.time()
    
    orchestrator = get_orchestrator()
    
    # Validate file_type
    valid_types = {"txt", "md", "html", "json", "xml"}
    if file_type not in valid_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file_type. Must be one of: {valid_types}"
        )
    
    # Ensure filename has correct extension
    if not filename.endswith(f".{file_type}"):
        filename = f"{filename}.{file_type}"
    
    try:
        config = IngestionConfig(create_sequential_edges=create_sequential_edges)
        
        result = orchestrator.ingest_file(
            filename=filename,
            content=text.encode('utf-8'),
            config=config
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return IngestionResponse(
            success=result.success,
            document_id=result.document_id,
            filename=result.filename,
            file_type=result.file_type,
            chunks_created=result.chunks_created,
            nodes_created=result.nodes_created,
            edges_created=result.edges_created,
            message=result.message,
            processing_time_ms=round(processing_time, 2),
            chunk_uuids=result.chunk_uuids
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error ingesting text")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
