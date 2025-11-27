import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# File Paths
SQLITE_DB_PATH = os.path.join(DATA_DIR, "sqlite.db")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
GRAPH_PATH = os.path.join(DATA_DIR, "graph.graphml")  # Changed from .gpickle to .graphml for robustness

# Embedding Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Search Defaults
DEFAULT_TOP_K = 10
DEFAULT_ALPHA = 0.7
DEFAULT_BETA = 0.3

# ============================================
# SCALABILITY CONFIGURATION
# ============================================

# FAISS Index Configuration
# Options: "flat" (brute-force, 100% recall) or "hnsw" (approximate, O(log N))
FAISS_INDEX_TYPE = os.environ.get("FAISS_INDEX_TYPE", "hnsw")

# HNSW Parameters (only used when FAISS_INDEX_TYPE="hnsw")
# M: Number of connections per layer (higher = more accurate, more memory)
# efConstruction: Size of dynamic list during construction (higher = better quality)
# efSearch: Size of dynamic list during search (higher = more accurate, slower)
HNSW_M = int(os.environ.get("HNSW_M", "32"))
HNSW_EF_CONSTRUCTION = int(os.environ.get("HNSW_EF_CONSTRUCTION", "200"))
HNSW_EF_SEARCH = int(os.environ.get("HNSW_EF_SEARCH", "64"))

# PageRank Configuration
# damping_factor: Probability of following a link (standard is 0.85)
# max_iter: Maximum iterations for convergence
# tol: Tolerance for convergence
PAGERANK_DAMPING = float(os.environ.get("PAGERANK_DAMPING", "0.85"))
PAGERANK_MAX_ITER = int(os.environ.get("PAGERANK_MAX_ITER", "100"))
PAGERANK_TOL = float(os.environ.get("PAGERANK_TOL", "1e-6"))

# Centrality cache settings
# How many seconds before PageRank is recalculated (0 = recalculate every query)
CENTRALITY_CACHE_TTL = int(os.environ.get("CENTRALITY_CACHE_TTL", "300"))  # 5 minutes
