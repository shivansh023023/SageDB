# SageDB: Vector + Graph Native Database

SageDB is a high-performance, hybrid retrieval system built for the hackathon. It combines the semantic power of **FAISS** (Vector Search) with the relational context of **NetworkX** (Graph Traversal), using **SQLite** as the atomic single source of truth.

## 🚀 Key Features

- **Twin-Engine Retrieval**: Seamlessly blends vector similarity and graph connectivity.
- **Late Interaction Fusion**: Configurable Alpha/Beta scoring for hybrid ranking.
- **Concurrency Safe**: Implements Reader-Writer locks for high-throughput read/write operations.
- **Atomic Consistency**: SQLite guarantees data integrity across vector and graph indices.
- **Production Ready**: Includes Pydantic validation, error handling, and benchmarks.

### 🆕 Scalability Features (v2.0)

| Feature                 | Description                                        | Benefit                                |
| ----------------------- | -------------------------------------------------- | -------------------------------------- |
| **HNSW Index**          | Hierarchical Navigable Small World graph for FAISS | O(log N) search vs O(N) brute-force    |
| **PageRank Centrality** | Graph importance scoring using PageRank algorithm  | Better node ranking than simple degree |
| **GraphML Persistence** | XML-based graph storage format                     | Human-readable, cross-platform, robust |
| **Centrality Caching**  | TTL-based cache for PageRank scores                | Faster repeated queries                |

## 🛠️ Tech Stack

- **Core**: Python 3.9+
- **API**: FastAPI, Uvicorn
- **Vector Engine**: FAISS (IndexHNSWFlat for scalability)
- **Graph Engine**: NetworkX with PageRank
- **Storage**: SQLite + GraphML
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)

## 📦 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/shivansh023023/SageDB.git
   cd SageDB
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the Server

Start the FastAPI server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

### Streamlit UI

```bash
streamlit run ui/app.py
```

The UI will be available at `http://localhost:8501`.

## ⚙️ Configuration

All scalability settings can be configured via environment variables or `config.py`:

```python
# FAISS Index Type: "flat" (brute-force) or "hnsw" (approximate)
FAISS_INDEX_TYPE = "hnsw"

# HNSW Parameters
HNSW_M = 32                    # Connections per layer
HNSW_EF_CONSTRUCTION = 200     # Build-time accuracy
HNSW_EF_SEARCH = 64            # Query-time accuracy

# PageRank Settings
PAGERANK_DAMPING = 0.85        # Standard damping factor
CENTRALITY_CACHE_TTL = 300     # Cache TTL in seconds (5 min)
```

## 📚 API Documentation

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Key Endpoints

- `POST /v1/nodes`: Create a new node (Atomic write to SQLite + FAISS + Graph).
- `POST /v1/edges`: Connect nodes with weighted relations.
- `POST /v1/search/hybrid`: Perform hybrid vector+graph search.
- `POST /v1/ingest/file`: Ingest documents with automatic chunking.
- `POST /v1/benchmark`: Run automated precision/recall metrics.

## 🧪 Architecture

```mermaid
graph LR
    User[User Input] --> Embed[Embedding Model]
    Embed --> SQLite[(SQLite\nSource of Truth)]
    SQLite --> FAISS[FAISS HNSW Index]
    SQLite --> Graph[NetworkX Graph\n+ PageRank]

    Query[Query] --> Embed
    Embed --> FAISS
    FAISS --> Candidates[Candidates]
    Candidates --> GraphScoring[PageRank Scoring]
    GraphScoring --> Fusion[Hybrid Fusion]
    Fusion --> Results[Final Results]

    Graph --> GraphML[(GraphML File)]
    FAISS --> Index[(FAISS Index File)]
```

## 🔬 Scalability Deep Dive

### HNSW Index

The Hierarchical Navigable Small World (HNSW) algorithm creates a multi-layer graph for approximate nearest neighbor search:

- **Complexity**: O(log N) vs O(N) for brute-force
- **Trade-off**: Slight recall loss (~99%) for massive speed gains
- **Best for**: 100K+ vectors

### PageRank Centrality

Instead of simple degree centrality, we use Google's PageRank algorithm:

- **Formula**: PR(A) = (1-d) + d × Σ(PR(T)/C(T))
- **Damping**: 0.85 (probability of following a link)
- **Benefit**: Identifies truly important nodes, not just connected ones

### GraphML Persistence

- Human-readable XML format
- Preserves all node/edge attributes
- No pickle security vulnerabilities
- Cross-platform compatible

## ⚖️ License

MIT License
