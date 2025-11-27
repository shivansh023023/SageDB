import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# File Paths
SQLITE_DB_PATH = os.path.join(DATA_DIR, "sqlite.db")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
GRAPH_PATH = os.path.join(DATA_DIR, "graph.gpickle")

# Embedding Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Search Defaults
DEFAULT_TOP_K = 10
DEFAULT_ALPHA = 0.7
DEFAULT_BETA = 0.3
