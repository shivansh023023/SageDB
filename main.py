import uvicorn
from fastapi import FastAPI
import logging
import os
import glob

from config import DATA_DIR
from api.routes import router
from ingestion.router import router as ingestion_router
from core.embedding import embedding_service
from storage.sqlite_ops import sqlite_manager
from storage.vector_ops import vector_index
from storage.graph_ops import graph_manager

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vector + Graph Native Database")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Vector + Graph Native Database...")
    
    # 1. Create Data Directory
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # 2. Cleanup Incomplete Snapshots
    for tmp_file in glob.glob(os.path.join(DATA_DIR, "*.tmp")):
        logger.warning(f"Removing incomplete snapshot: {tmp_file}")
        os.remove(tmp_file)
        
    # 3. Initialize Embedding Service & Warmup
    embedding_service.warmup()
    
    # 4. Initialize Storage
    # SQLite is initialized on import/instantiation
    
    # 5. Load/Rebuild Indices
    # Load all nodes from SQLite to rebuild ID map
    logger.info("Rebuilding in-memory ID map from SQLite...")
    id_map = sqlite_manager.get_all_nodes_map()
    
    # Load FAISS
    vector_index.load_or_create(initial_id_map=id_map)
    
    # Load Graph
    graph_manager.load_or_create()
    
    # 6. Integrity Check
    sqlite_count = sqlite_manager.count_nodes()
    faiss_count = vector_index.ntotal
    
    logger.info(f"Integrity Check - SQLite Nodes: {sqlite_count}, FAISS Vectors: {faiss_count}")
    
    if sqlite_count != faiss_count:
        logger.warning("MISMATCH DETECTED: SQLite and FAISS counts do not match. Rebuilding FAISS index...")
        # Rebuild FAISS index from SQLite
        rebuild_faiss_from_sqlite()
    
    logger.info("System Ready.")


def rebuild_faiss_from_sqlite():
    """Rebuild FAISS index from SQLite data."""
    logger.info("Starting FAISS index rebuild from SQLite...")
    
    # Get all nodes from SQLite
    all_nodes = sqlite_manager.get_all_nodes(limit=10000, offset=0)
    
    for node in all_nodes:
        uuid = node['uuid']
        text = node['text']
        faiss_id = node['faiss_id']
        
        # Generate embedding
        vector = embedding_service.encode(text)
        
        # Add to FAISS
        vector_index.add_vector(vector, faiss_id, uuid)
        
        # Also ensure node is in graph
        graph_manager.add_node(uuid)
    
    # Rebuild edges in graph from SQLite
    all_edges = sqlite_manager.get_all_edges(limit=10000, offset=0)
    for edge in all_edges:
        graph_manager.add_edge(
            edge['source_id'], 
            edge['target_id'], 
            edge['relation'], 
            edge['weight']
        )
    
    logger.info(f"FAISS rebuild complete. Total vectors: {vector_index.ntotal}")


app.include_router(router)
app.include_router(ingestion_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
