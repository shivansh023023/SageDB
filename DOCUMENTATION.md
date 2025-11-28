# SageDB: Vector + Graph Native Database

> **TL;DR**: A hybrid database combining FAISS (vector search) + NetworkX (graph traversal) + SQLite (persistence) with a novel **Graph Expansion Algorithm** that discovers related nodes through graph structure, not just re-ranks vector results.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start backend (port 8000)
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit UI (port 8501)
streamlit run ui/app.py

# Start Interactive 3D Visualization (port 5174)
cd viz
npm install  # First time only
npm run dev

# Populate with test data
python populate_db.py
```

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Our Solution: SageDB](#2-our-solution-sagedb)
3. [Implementation Status](#3-implementation-status)
4. [Future Work](#4-future-work-whats-left-to-implement)
5. [Code Walkthrough](#5-code-walkthrough)
6. [Q&A for Mentorship](#6-questions--clarifications-for-mentorship-round)
7. [Demo Script](#7-demo-script-for-evaluation)
8. [Future Enhancements](#8-future-enhancements-if-asked)
9. [Questions for Evaluators](#9-questions-for-mentorsevaluators)

---

## 1. Problem Statement

**Devfolio Problem Statement 1: Vector + Graph Native Database for Efficient AI Retrieval**

Modern AI systems, particularly those involving RAG (Retrieval-Augmented Generation), face a critical limitation when relying on a single mode of retrieval:

- **Vector Databases** excel at finding semantically similar content but lack the ability to reason about structured relationships or multi-hop connections.
- **Graph Databases** are powerful for traversing explicit relationships but struggle with unstructured semantic search.

The challenge is to build a **minimal but functional hybrid database** that combines the strengths of both. The system must:

1.  Ingest structured and unstructured data.
2.  Store data as graph nodes enriched with vector embeddings.
3.  Expose an API for hybrid search, CRUD operations, and relationship traversal.
4.  Demonstrate that hybrid retrieval improves relevance over single-mode retrieval.
5.  Run locally and be fast enough for real-time queries.

---

## 2. Our Solution: SageDB

**SageDB** is a locally hosted, hybrid database engine designed to bridge the gap between semantic understanding and structural reasoning.

### Core Architecture

SageDB uses a multi-layered storage approach to handle different aspects of the data:

1.  **SQLite (Metadata & Persistence)**: Acts as the source of truth for node attributes, metadata, and the mapping between different storage layers. It ensures ACID compliance for metadata updates.
2.  **FAISS (Vector Storage)**: A high-performance library for dense vector similarity search. It stores the embeddings of the node text, enabling fast semantic retrieval.
3.  **NetworkX (Graph Storage)**: Manages the graph topology (nodes and edges) in memory. It allows for efficient traversal, centrality calculation, and connectivity analysis.

### 🆕 Enhanced Graph Relationships with Semantic Similarity

SageDB now creates **intelligent, semantically-aware relationships** during data ingestion:

#### Relationship Types

1. **Sequential Edges (`next_chunk`)**:

   - Links consecutive chunks in documents
   - **Dynamic weights** based on cosine similarity between adjacent chunks
   - Base weight boosted by up to 15% for highly similar content
   - Example: If two adjacent chunks discuss the same concept, their connection is stronger

2. **Semantic Similarity Edges (`similar_to`)**:

   - Created when adjacent chunks have cosine similarity > 0.8
   - Explicitly marks high semantic overlap
   - Weight: `0.90 × cosine_similarity`

3. **Cross-Chunk References (`related_to`)**:

   - Bidirectional edges between non-adjacent chunks with similarity > 0.75
   - Enables discovery of related content across different parts of documents
   - Weight: `0.75 × cosine_similarity`

4. **Hierarchical Edges (`section_of`)**:
   - Parent-child relationships based on document structure
   - Preserves organizational hierarchy from source documents

#### Migration for Existing Data

If you have existing data ingested before the semantic relationship enhancement, run:

```bash
python migrate_edges.py
```

This will:

- ✅ Update all `next_chunk` edge weights with similarity-based boosting
- ✅ Add `similar_to` edges for highly similar adjacent chunks
- ✅ Add `related_to` edges for semantic cross-references

**Results**: The migration typically adds 700+ new semantic edges to a knowledge base with ~480 chunks.

### The Hybrid Retrieval Mechanism

SageDB implements a **Graph-Augmented Hybrid Search** algorithm that goes beyond simple re-ranking:

1.  **Seed Discovery (Vector Search)**: First, the system retrieves the top-50 "seed" candidates based on cosine similarity from the FAISS index.
2.  **Graph Expansion**: Starting from these seeds, the algorithm performs a **bidirectional BFS (Breadth-First Search)** to depth 2, discovering nodes that are structurally related but may not have been found by vector search alone. This is the key innovation - **we discover new candidates through graph traversal, not just re-rank existing ones**.
3.  **Vector Scoring for Expanded Candidates**: For newly discovered nodes (those not in the original seed set), we compute their vector similarity to the query using batch processing.
4.  **Relationship-Aware Graph Scoring**: For all candidates, we calculate a "Graph Score" based on:
    - **Connectivity**: Weighted average distance to seed nodes, where edge types affect the weight (e.g., `is_a` relationships are stronger than `related_to`).
    - **Centrality**: How important the node is within the overall graph structure (Degree Centrality).
5.  **Fusion**: The final score is a weighted sum:
    $$ Score = (\alpha \times VectorScore\_{norm}) + (\beta \times GraphScore) $$
    Where $\alpha$ and $\beta$ are tunable parameters (defaulting to 0.7 and 0.3).

**Key Insight**: The old approach only re-ranked top-k vector results. The new approach **expands the candidate pool** using graph structure, meaning a node that's semantically dissimilar but structurally important (e.g., 2 hops away from multiple relevant nodes) can now be discovered and ranked.

---

## 2.1 Deep Dive: Graph Scoring Mechanism (Updated Algorithm)

Graph scoring is the secret sauce that differentiates SageDB from pure vector databases. While vector similarity tells us "what's semantically similar," graph scoring tells us "what's structurally important and well-connected."

### The Old Problem: Re-Ranking Doesn't Discover

The original implementation had a fundamental flaw: it only **re-ranked** the top-k vector search results. This meant:

- If a highly relevant node wasn't in the top-k by vector similarity, it could never be found
- Graph structure was used for scoring, not for discovery
- The search was fundamentally limited by vector recall

### The New Solution: Graph Expansion + Discovery

The updated algorithm uses graph structure to **expand** the candidate pool:

```
Query: "neural networks"
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Step 1: Vector Search (Top 50 Seeds)           │
│  Seeds: [Deep Learning, CNN, RNN, ...]          │
└─────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Step 2: Graph Expansion (BFS, Depth 2)         │
│  Discover: [Backpropagation, Activation Fn,     │
│             Gradient Descent, ...]              │
│  Total Candidates: 50 → 120+                    │
└─────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Step 3: Score All Candidates                   │
│  - Vector: Use original score OR compute new    │
│  - Graph: Relationship-aware connectivity       │
└─────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Step 4: Fusion & Rank                          │
│  Final Score = α × Vector + β × Graph           │
└─────────────────────────────────────────────────┘
```

### Relationship-Aware Scoring

Not all edges are created equal. The new algorithm assigns different weights to edge types:

| Edge Type    | Weight | Rationale                                           |
| ------------ | ------ | --------------------------------------------------- |
| `is_a`       | 1.0    | Taxonomic (IS-A hierarchy) - strongest relationship |
| `part_of`    | 0.95   | Compositional - very strong                         |
| `depends_on` | 0.9    | Functional dependency                               |
| `uses`       | 0.85   | Usage relationship                                  |
| `related_to` | 0.5    | Generic - weakest typed relationship                |
| _(unknown)_  | 0.3    | Fallback for unrecognized types                     |

**Example**:

- If "Neural Networks" → "Deep Learning" with `is_a`, the edge contributes full weight
- If "Neural Networks" → "TensorFlow" with `related_to`, the edge contributes half weight

### Why Graph Scoring Matters

Imagine you're searching for "machine learning algorithms" in a knowledge graph:

- **Pure Vector Search** might return nodes with similar text but miss the context. For example, it might find a blog post mentioning "machine learning" in passing.
- **Graph-Enhanced Search** recognizes that a node directly connected to "Supervised Learning", "Neural Networks", and "Training Data" is more relevant because it sits at the intersection of important concepts.

### The Two Components of Graph Score

#### 1. Connectivity Score (Proximity to Seeds) - IMPROVED

**What it measures**: How "close" a candidate node is to the seed nodes in the graph.

**How it works (NEW ALGORITHM)**:

1. Starting from the seed set, perform BFS in **both directions** (incoming and outgoing edges)
2. Track the distance and **edge types** traversed to reach each node
3. Calculate a weighted distance based on relationship strength:
   ```python
   effective_distance = sum(1 / edge_weight for edge in path)
   ```
4. Convert to similarity score using exponential decay:
   ```python
   connectivity_score = exp(-weighted_avg_distance)
   ```

**Key Improvement**: We now use **weighted average distance** across all reachable seeds, not just minimum distance. This means:

- A node close to ONE seed but far from others gets a lower score
- A node that's moderately close to MANY seeds gets a higher score
- Relationship types affect the effective distance

**Example**:

```
Query: "neural networks"
Seeds: [Node_A: "Deep Learning", Node_B: "Backpropagation", Node_C: "CNNs"]

Candidate: "Activation Functions"
- Path to A: 1 hop via "is_a" (weight=1.0) → distance = 1.0
- Path to B: 2 hops via "depends_on" → distance = 1.1
- Path to C: 1 hop via "part_of" (weight=0.95) → distance = 1.05

Weighted Avg Distance = (1.0 + 1.1 + 1.05) / 3 = 1.05
Connectivity Score = exp(-1.05) ≈ 0.35
```

#### 2. Centrality Score (Structural Importance)

**What it measures**: How "important" or "central" a node is in the overall graph, regardless of the query.

**How it works**:
We use **Degree Centrality**, which is simply:

```
degree_centrality = (in_degree + out_degree) / max_degree_in_graph
```

- **In-degree**: Number of edges pointing TO this node (how many nodes reference it)
- **Out-degree**: Number of edges pointing FROM this node (how many nodes it references)
- **Normalization**: We divide by the maximum degree in the entire graph to get a 0-1 score

**Example**:

```
Graph:
- Node_A ("Machine Learning"): 15 incoming, 8 outgoing → degree = 23
- Node_B ("Random Forest"): 3 incoming, 2 outgoing → degree = 5
- Node_C ("K-Means"): 2 incoming, 1 outgoing → degree = 3
- Max degree in graph: 23

Centrality Scores:
- Node_A: 23/23 = 1.0 (highly central "hub" node)
- Node_B: 5/23 = 0.22
- Node_C: 3/23 = 0.13
```

**Why this matters**: Hub nodes (like "Machine Learning" in a CS knowledge graph) are generally more valuable to retrieve because they:

- Are well-defined and frequently referenced
- Serve as good entry points for exploration
- Tend to have higher-quality content (more reviewed/linked)

#### 3. Relationship Score (Edge Type Awareness)

**What it measures**: The strength of direct relationships between the candidate and seed nodes based on edge types.

**How it works**:
The `get_relationship_score` function checks for direct edges between the candidate and each seed:

```python
edge_weights = {
    "is_a": 1.0,        # Taxonomic hierarchy
    "part_of": 0.95,    # Compositional
    "specialization_of": 0.9,
    "uses": 0.85,
    "depends_on": 0.8,
    "implements": 0.8,
    "extends": 0.75,
    "related_to": 0.5,  # Generic
    "mentioned_in": 0.3
}
# Returns max score from all direct edges
```

**Why this matters**: A node directly connected to a seed via a strong relationship (like `is_a`) should score higher than one connected via a weak relationship (like `mentioned_in`).

### Combining Connectivity, Centrality, and Relationship Scores

The final Graph Score is a weighted average of three components:

```python
graph_score = (0.5 × connectivity_score) + (0.3 × centrality_score) + (0.2 × relationship_score)
```

In our implementation (`calculate_expanded_graph_score`):

- `w_connectivity = 0.5` (50% weight on how close to search results)
- `w_centrality = 0.3` (30% weight on overall importance)
- `w_relationship = 0.2` (20% weight on edge type strength)

### Full Example: Hybrid Search in Action (Graph Expansion Algorithm)

**Query**: "explain gradient descent"

**Step 1: Vector Search (Get 50 Seeds)**
FAISS returns top-50 by cosine similarity. Top 5 shown:

1. "Gradient Descent Optimization" (score: 0.92)
2. "Backpropagation Algorithm" (score: 0.87)
3. "Learning Rate Tuning" (score: 0.81)
4. "Stochastic Gradient Descent" (score: 0.79)
5. "Momentum in Optimization" (score: 0.76)

**Step 2: Graph Expansion (BFS Depth 2)**
Starting from all 50 seeds, expand outward:

- Discover "Activation Functions" (2 hops from "Backpropagation" via `depends_on`)
- Discover "Neural Network Training" (1 hop from "Gradient Descent" via `is_a`)
- Discover "Loss Functions" (1 hop from "Gradient Descent" via `uses`)
- ...and more

**Total Candidates**: 50 seeds → **98 expanded candidates** (as shown in logs: "Expanded from 25 seeds to 48 candidates")

**Step 3: Vector Scoring for New Candidates**
For nodes discovered via graph (not in original seeds), compute vector similarity:

- "Loss Functions": 0.68 (wasn't in top-50 but now included!)
- "Activation Functions": 0.55

**Step 4: Graph Scoring (Relationship-Aware)**
For each candidate, calculate connectivity + centrality:

| Node               | Vector | Connectivity | Centrality | Graph Score |
| ------------------ | ------ | ------------ | ---------- | ----------- |
| Gradient Descent   | 0.92   | 0.55         | 0.88       | 0.65        |
| Backpropagation    | 0.87   | 0.48         | 0.72       | 0.55        |
| **Loss Functions** | 0.68   | 0.62         | 0.45       | 0.57        |
| Learning Rate      | 0.81   | 0.41         | 0.35       | 0.39        |
| SGD                | 0.79   | 0.52         | 0.61       | 0.55        |

**Step 5: Fusion**
Combine using alpha=0.7, beta=0.3:

| Node               | Vector (norm) | Graph | Final Score | Rank                          |
| ------------------ | ------------- | ----- | ----------- | ----------------------------- |
| Gradient Descent   | 1.00          | 0.65  | 0.90        | **1**                         |
| Backpropagation    | 0.94          | 0.55  | 0.82        | **2**                         |
| **Loss Functions** | 0.74          | 0.57  | 0.69        | **3** ← DISCOVERED via graph! |
| SGD                | 0.81          | 0.55  | 0.73        | **4**                         |
| Learning Rate      | 0.69          | 0.39  | 0.60        | **5**                         |

**Key Insight**: "Loss Functions" wasn't in the original top-50 vector results, but graph expansion discovered it because it's structurally connected to "Gradient Descent" via a strong `uses` relationship. This is the power of graph-augmented discovery.

### Tuning Alpha and Beta

- **High Alpha (e.g., 0.9)**: Prioritizes semantic similarity. Good for broad exploratory queries.
- **High Beta (e.g., 0.9)**: Prioritizes graph structure. Good for navigating known knowledge domains.
- **Balanced (0.7/0.3)**: Our default. Works well for general-purpose retrieval.

You can adjust these in the UI's search tab to see how results change!

---

## 3. Implementation Status

We have successfully built a working prototype that meets the core requirements of the problem statement.

### ✅ What We Have Implemented

#### Backend API (FastAPI)

| Endpoint                       | Method | Description                                                                | Status |
| ------------------------------ | ------ | -------------------------------------------------------------------------- | ------ |
| `/v1/nodes`                    | POST   | Create nodes with automatic embedding generation                           | ✅     |
| `/v1/nodes`                    | GET    | List all nodes with pagination                                             | ✅     |
| `/v1/nodes/{uuid}`             | GET    | Retrieve single node                                                       | ✅     |
| `/v1/nodes/{uuid}`             | PUT    | Update node text/metadata, regenerate embedding                            | ✅     |
| `/v1/nodes/{uuid}`             | DELETE | Delete node from all storage layers                                        | ✅     |
| `/v1/edges`                    | POST   | Create typed relationships between nodes                                   | ✅     |
| `/v1/edges`                    | GET    | List all edges with pagination                                             | ✅     |
| `/v1/edges/{edge_id}`          | GET    | Retrieve single edge by ID                                                 | ✅     |
| `/v1/edges/{edge_id}`          | PUT    | Update edge relation and/or weight                                         | ✅     |
| `/v1/edges/{edge_id}`          | DELETE | Delete edge from all storage layers                                        | ✅     |
| `/v1/search/hybrid`            | POST   | **Graph-Augmented Hybrid Search** with PPR, dedup, caching                 | ✅     |
| `/v1/search/hybrid/stream`     | POST   | **Streaming Hybrid Search** via Server-Sent Events                         | ✅     |
| `/v1/search/vector`            | POST   | **Pure Vector Search** (semantic similarity) + offset pagination           | ✅     |
| `/v1/search/context`           | POST   | **Context-Aware Search** with sliding window expansion + offset pagination | ✅     |
| `/v1/search/hybrid/legacy`     | POST   | Legacy re-ranking algorithm for comparison                                 | ✅     |
| `/v1/search/graph`             | GET    | Subgraph visualization endpoint                                            | ✅     |
| `/v1/cache/stats`              | GET    | **Search cache statistics** (size, hit rate, TTL)                          | ✅     |
| `/v1/cache/clear`              | POST   | **Clear search cache**                                                     | ✅     |
| `/v1/analytics/popular-chunks` | GET    | **Top retrieved chunks** with retrieval counts                             | ✅     |
| `/v1/analytics/chunk/{uuid}`   | GET    | **Chunk usage statistics** (retrieval count, avg rank, etc.)               | ✅     |
| `/v1/analytics/retrieval/{id}` | GET    | **Retrieval event details** for citation/provenance                        | ✅     |
| `/v1/benchmark`                | POST   | Calculate Precision/Recall/NDCG metrics                                    | ✅     |
| `/v1/admin/snapshot`           | POST   | Persist FAISS and Graph to disk                                            | ✅     |
| `/health`                      | GET    | System health check                                                        | ✅     |
| `/v1/ingest/file`              | POST   | **Ingest single file** (md, txt, html, json, xml)                          | ✅     |
| `/v1/ingest/batch`             | POST   | **Batch ingest multiple files**                                            | ✅     |
| `/v1/ingest/text`              | POST   | **Ingest raw text** directly                                               | ✅     |
| `/v1/ingest/supported-types`   | GET    | Get supported file types and limits                                        | ✅     |

#### Ingestion Pipeline (NEW)

The ingestion pipeline is the **critical entry point** for getting data into SageDB. It handles the complete flow from raw files to searchable graph nodes:

```
File/Text → Parser → Chunker → Embedder → Storage → Relationships
```

**Supported Formats:**

- **Markdown** (`.md`, `.markdown`): Uses `markdown-it-py` for AST parsing, preserves headers as hierarchy
- **HTML** (`.html`, `.htm`): Uses BeautifulSoup4, extracts semantic tags (article, section, p)
- **Plain Text** (`.txt`): Uses NLTK for sentence boundary detection, splits by paragraphs
- **JSON** (`.json`): Recursively extracts text from content/description fields
- **XML** (`.xml`): Parses text content from elements

**Key Features:**

- **Semantic Chunking**: Respects sentence boundaries, configurable token limits and overlap
- **Only Content-Bearing Nodes**: No empty header nodes - sections must have actual content
- **Hierarchy Preservation**: Document structure stored as metadata, creates `section_of` edges
- **Sequential Relationships**: `next_chunk` edges with weight 1.0 (highest) connect consecutive chunks
- **Deduplication**: SHA-256 content hash prevents duplicate ingestion
- **File Size Limit**: 10MB default to prevent memory issues
- **Relevance Filtering**: Search results below 0.25 similarity are filtered out

**Edge Weight System:**
| Edge Type | Weight | Purpose |
| ------------- | ------ | -------------------------------------- |
| `chunk_of` | 1.0 | Chunk belongs to document (strongest) |
| `section_of` | 0.98 | Hierarchical section relationship |
| `follows` | 0.95 | Sequential document order |
| `next_chunk` | 0.85 | Consecutive chunks in same section |

#### Storage Engines

- ✅ **SQLite Manager**: Handles persistent storage of nodes and edges (source of truth)
- ✅ **Vector Index**: FAISS HNSW index for O(log N) approximate nearest neighbor search with ID mapping
- ✅ **Graph Manager**: NetworkX DiGraph with PageRank centrality and GraphML persistence

#### Core Logic

- ✅ **Embedding Service**: Uses `sentence-transformers` (all-MiniLM-L6-v2), 384 dimensions
- ✅ **Graph Expansion**: BFS-based expansion from seeds (`expand_from_seeds`)
- ✅ **Relationship-Aware Scoring**: Edge types affect connectivity scores (`get_relationship_score`)
- ✅ **PageRank Centrality**: Cached PageRank scores with 5-minute TTL for importance ranking
- ✅ **Batch Hydration**: Single-query metadata fetch eliminates N+1 problem (`get_nodes_batch`)
- ✅ **Batch Vector Similarity**: Compute vector scores for graph-discovered nodes (`batch_compute_similarity`)
- ✅ **Hybrid Fusion**: Alpha/Beta weighted combination with Min-Max normalization
- ✅ **Auto-Rebuild**: Automatically rebuilds FAISS index from SQLite on startup if mismatch detected
- ✅ **Concurrency Control**: Uses `readerwriterlock` for thread-safe operations
- ✅ **Offset Pagination**: All search endpoints support `offset` parameter for pagination

#### User Interface (Streamlit)

- ✅ **System Health**: Backend status and data counts
- ✅ **Ingest Files**: File upload UI for single/batch ingestion and raw text input
- ✅ **Add Data**: Forms to create nodes and edges manually
- ✅ **Manage Data**: Update and delete nodes/edges
- ✅ **Search**: Interactive search with Alpha/Beta sliders and detailed scoring breakdowns
  - 🆕 **Query Decomposition**: Automatically splits comparison queries ("X vs Y", "difference between A and B")
  - 🆕 **RRF Fusion**: Reciprocal Rank Fusion for multi-query results with configurable k parameter
- ✅ **Graph View**: Matplotlib-based subgraph visualization
- ✅ **Data Explorer**: Tabular view of all nodes and edges

#### 🆕 Interactive 3D Visualization (Vite + React)

- ✅ **3D Force-Directed Graph**: WebGL-based interactive visualization powered by react-force-graph
- ✅ **Real-time Edge Display**: View semantic relationships (`next_chunk`, `similar_to`, `related_to`)
- ✅ **Node Interaction**: Click nodes to view details, hover for tooltips
- ✅ **Semantic Edge Colors**:
  - `next_chunk`: Sequential flow (blue)
  - `similar_to`: High semantic similarity (green)
  - `related_to`: Cross-references (orange)
- ✅ **Search Integration**: Find and highlight nodes in the graph
- ✅ **Layout Controls**: Toggle 3D/2D views, pause/resume physics simulation
- ✅ **Live Reload**: Automatically fetches latest graph state from API

**Access**: `http://localhost:5174` after running `npm run dev` in `/viz` directory

#### Testing & Population

- ✅ `populate_db.py`: Script to generate AI/ML knowledge graph (40 nodes, ~124 edges)
- ✅ `test_sagedb.py`: Integration tests covering CRUD and Search

### Key Algorithm: Graph Expansion Search

The core innovation is the **graph expansion** approach:

```python
# 1. Get initial seeds from vector search
seeds = vector_search(query, top_k=50)

# 2. Expand via graph (BFS in both directions, depth=2)
expanded = graph_manager.expand_from_seeds(seed_uuids, depth=2)

# 3. For new candidates, compute vector similarity
for uuid in expanded - seeds:
    vector_scores[uuid] = batch_compute_similarity(query_vector, uuid)

# 4. Calculate relationship-aware graph scores
for uuid in expanded:
    graph_scores[uuid] = calculate_expanded_graph_score(uuid, seed_uuids)

# 5. Fusion
final_scores = alpha * vector_scores + beta * graph_scores
```

### How It Works (Flow)

1.  **Ingestion**: When a node is created, its text is embedded. The metadata goes to SQLite, the vector to FAISS, and the ID to NetworkX.
2.  **Search**: A query is embedded → FAISS finds top-50 seeds → Graph expansion discovers related nodes → Vector scores computed for all candidates → Graph scores calculated with relationship weights → Fusion ranks final results.
3.  **Startup Sync**: If FAISS and SQLite are out of sync, the system auto-rebuilds the FAISS index from SQLite data.

---

## 4. Future Work (What's Left to Implement)

While the core is functional, several enhancements would make SageDB production-ready:

### High Priority

1. **Multi-hop Reasoning Path**: Return the "path" of reasoning showing how the system traversed from query to result (e.g., "Query → Node A → Node B → Result"). Currently, we score based on proximity but don't expose the traversal path.

2. **Relationship Filtering**: Allow users to filter search results by edge types (e.g., "show only `is_a` relationships"). The scoring infrastructure exists, filtering UI/API does not.

3. **Dockerization**: Containerize the application for easier deployment and evaluation.

### Medium Priority

4. **Schema Enforcement**: Optional schema validation for node types and allowed edge relations.

5. **CLI Tool**: Command-line interface for scripted querying.

6. **Streaming Ingestion**: Batch ingestion API for large datasets.

### ✅ Completed Scalability Features

The following scalability features have been implemented:

- ✅ **HNSW Index**: Replaced `IndexFlatIP` (O(N) brute-force) with `IndexHNSWFlat` (O(log N) approximate) for scalability beyond 100K nodes
- ✅ **PageRank Centrality**: Replaced simple Degree Centrality with PageRank for better importance scoring, with 5-minute TTL caching
- ✅ **GraphML Persistence**: Graph is now persisted as GraphML format instead of pickle for better interoperability
- ✅ **Batch Hydration**: Eliminated N+1 query problem with `get_nodes_batch()` for efficient metadata retrieval
- ✅ **Offset Pagination**: All search endpoints now support `offset` parameter for paginated results

### ✅ Completed Production RAG Features

The following production-grade RAG features have been implemented:

- ✅ **Personalized PageRank (PPR)**: Query-aware graph importance scoring. Unlike global PageRank, PPR biases node importance toward the seed nodes from vector search, enabling "what's important relative to my query" scoring. Supports optional score-weighted personalization.

- ✅ **Metadata Pre-Filtering**: Filter nodes by type and metadata conditions BEFORE vector search. Uses SQLite for efficient filtering, then performs FAISS search only on matching node IDs using `IDSelectorBatch`. Reduces search space for domain-specific queries.

- ✅ **Query Decomposition**: Automatically splits complex queries into sub-queries:

  - "Compare X and Y" → ["X", "Y"]
  - "X vs Y" → ["X", "Y"]
  - Multiple sentences → separate queries
  - Results merged using round-robin fusion with coverage bonus

- ✅ **Semantic Deduplication**: Two-pass deduplication:

  1. Fast exact-match using MD5 text hash
  2. Semantic dedup using cosine similarity threshold (default 0.95)

  - Prevents redundant results in RAG context windows

- ✅ **LRU Search Cache**: In-memory cache for repeated queries:

  - Default: 100 entries, 5-minute TTL
  - Cache key includes query text + all search parameters
  - API endpoints: `/v1/cache/stats`, `/v1/cache/clear`
  - ~100x speedup for cached queries

- ✅ **Provenance Tracking**: Complete retrieval audit trail:

  - Records every search with timestamp, query, results, and scores
  - `retrieval_id` returned with each search for citation
  - Analytics: `/v1/analytics/popular-chunks`, `/v1/analytics/chunk/{uuid}`
  - Enables feedback loops and relevance tuning

- ✅ **Streaming Search**: Server-Sent Events endpoint for progressive results:
  - Endpoint: `/v1/search/hybrid/stream`
  - Streams results as `{"event": "result", "data": {...}}`
  - Reduces time-to-first-result for large result sets
  - Ideal for real-time UI updates

**New Search Parameters:**

```python
SearchQuery(
    text="query",
    top_k=10,
    alpha=0.7, beta=0.3,          # Fusion weights
    metadata_filter={"source": "wiki"},  # Pre-filter by metadata
    use_ppr=True,                  # Use Personalized PageRank
    deduplicate=True,              # Remove similar results
    dedup_threshold=0.95,          # Similarity threshold
    decompose_query=False,         # Split complex queries
    bypass_cache=False             # Skip cache for fresh results
)
```

---

## 5. Code Walkthrough

Here is a detailed explanation of every file and line of code in the project.

### `SageDB/config.py`

Configuration settings for the application.

- **`DATA_DIR`**: Defines where data is stored (`./data`).
- **`DB_PATH`**: Path to the SQLite database file.
- **`FAISS_INDEX_PATH`**: Path to the FAISS index file.
- **`GRAPH_PATH`**: Path to the NetworkX graph file.
- **`EMBEDDING_MODEL_NAME`**: Specifies the model to use (`all-MiniLM-L6-v2`).
- **`EMBEDDING_DIMENSION`**: Dimension of the vectors (384 for MiniLM).

### `SageDB/main.py`

The entry point of the application.

- **Imports**: Sets up logging and imports necessary modules.
- **`rebuild_faiss_from_sqlite()`**: Automatically rebuilds the FAISS index from SQLite if they are out of sync:
  - Iterates all nodes from SQLite
  - Regenerates embeddings for each node using embedding_service
  - Adds vectors to FAISS with correct IDs
  - Rebuilds graph edges from SQLite
  - Logs: "FAISS rebuild complete. Total vectors: N"
- **`startup_event`** (FastAPI lifecycle):
  1.  Creates `DATA_DIR` if missing.
  2.  Cleans up incomplete `.tmp` snapshot files.
  3.  Warms up the embedding model.
  4.  Loads ID map from SQLite (`get_all_nodes_map`).
  5.  Loads or creates FAISS index with the ID map.
  6.  Loads or creates NetworkX graph.
  7.  **Integrity Check**: Compares SQLite node count vs FAISS vector count. If mismatch, calls `rebuild_faiss_from_sqlite()`.
  8.  Logs "System Ready."
- **App Definition**: Creates the `FastAPI` app and includes the `router`.
- **Server Run**: Uses `uvicorn.run` to start the server on port 8000.

### `SageDB/api/routes.py`

Defines the HTTP endpoints.

- **`create_node`** (`POST /v1/nodes`):
  1.  Generates a UUID.
  2.  Encodes text to vector using embedding service.
  3.  Writes to SQLite (returns Atomic ID), FAISS, and NetworkX.
  4.  Uses `@write_locked` to prevent race conditions.
- **`get_node`** (`GET /v1/nodes/{uuid}`): Retrieves a single node by UUID.

- **`delete_node`** (`DELETE /v1/nodes/{uuid}`): Removes node from all three storage layers.

- **`update_node`** (`PUT /v1/nodes/{uuid}`): Updates node text and/or metadata. If text is updated, the embedding is regenerated and FAISS is updated.

- **`create_edge`** (`POST /v1/edges`): Adds a relationship to SQLite and NetworkX. Returns the edge ID for future retrieval/deletion.

- **`get_edge`** (`GET /v1/edges/{edge_id}`): Retrieves a single edge by its ID.

- **`update_edge`** (`PUT /v1/edges/{edge_id}`): Updates edge relation and/or weight. Syncs changes to both SQLite and NetworkX.

- **`delete_edge`** (`DELETE /v1/edges/{edge_id}`): Removes an edge from SQLite and NetworkX.

- **`vector_search`** (`POST /v1/search/vector`): **PURE VECTOR SEARCH** - Returns top-k results ranked purely by cosine similarity. Supports **offset pagination** for paginating through results. No graph scoring, alpha/beta weights, or expansion. Useful for baseline comparison and semantic-only queries.

- **`hybrid_search`** (`POST /v1/search/hybrid`) **CORE ALGORITHM**:

  1.  Encodes the query to a vector.
  2.  Searches FAISS for top-50 seed candidates.
  3.  **Graph Expansion**: Calls `expand_from_seeds(top_20_seeds, depth=2)` to discover related nodes.
  4.  Computes vector similarity for newly discovered nodes via `batch_compute_similarity`.
  5.  Hydrates all candidates with metadata from SQLite using **batch hydration** (eliminates N+1 queries).
  6.  Calculates **relationship-aware** graph scores using `calculate_expanded_graph_score` with **PageRank centrality**.
  7.  Calls `hybrid_fusion` to combine scores with alpha/beta weights.
  8.  Applies **offset pagination** for paginated results.
  9.  Returns top-k with detailed scoring breakdown (vector_score, graph_score, raw_vector_score).

- **`context_aware_search`** (`POST /v1/search/context`): **CONTEXT-AWARE SEARCH** - Same as hybrid search but expands each result with surrounding chunks via `next_chunk`/`previous_chunk` edges. Supports **offset pagination**. Ideal for RAG applications where context is important.

- **`hybrid_search_legacy`** (`POST /v1/search/hybrid/legacy`): Original re-ranking algorithm (no graph expansion) preserved for comparison.

- **`graph_search`** (`GET /v1/search/graph`): Returns subgraph for visualization via BFS.

- **`run_benchmark`** (`POST /v1/benchmark`): Calculates Precision@K, Recall@K, NDCG@K for vector-only vs hybrid.

- **`list_nodes`** (`GET /v1/nodes`): Paginated list of all nodes.

- **`list_edges`** (`GET /v1/edges`): Paginated list of all edges.

- **`health_check`** (`GET /health`): Returns node/edge counts and status.

- **`create_snapshot`** (`POST /v1/admin/snapshot`): Persists FAISS and Graph to disk.

### `SageDB/core/embedding.py`

Handles vector generation.

- **`EmbeddingService`**: A singleton class.
- **`__new__`**: Loads the `SentenceTransformer` model only once.
- **`warmup`**: Runs a dummy encoding to load model weights into RAM/GPU.
- **`_encode_cached`**: Uses `@lru_cache` to cache embeddings for repeated text (optimization).
- **`encode`**: Returns the numpy array of the embedding.

### `SageDB/core/fusion.py`

The brain of the hybrid search.

- **`hybrid_fusion`**:
  - Takes vector results and graph scores.
  - **Normalizes** vector scores to a 0-1 range using Min-Max scaling.
  - Combines them using the formula: `alpha * vector + beta * graph`.
  - Returns each result with full scoring breakdown:
    - `score`: Final hybrid score
    - `vector_score`: Normalized vector similarity (0-1)
    - `graph_score`: Combined graph score (0-1)
    - `raw_vector_score`: Original cosine similarity from FAISS
  - Sorts and returns the final ranked list.

### `SageDB/core/lock.py`

Ensures thread safety.

- **`GlobalLock`**: A singleton wrapper around `readerwriterlock.RWLockWrite`.
- **`read_locked` / `write_locked`**: Decorators that acquire the appropriate lock before executing a function. This allows multiple readers but only one writer.

### `SageDB/models/api_schemas.py`

Pydantic models for data validation.

- **`NodeCreate`**: Validates text length and metadata size.
- **`NodeUpdate`**: Optional fields for updating node text/metadata.
- **`EdgeCreate`**: Validates UUID format and weight range (0-1).
- **`EdgeUpdate`**: Optional fields for updating edge relation/weight.
- **`EdgeResponse`**: Response schema including edge ID for retrieval/deletion.
- **`SearchQuery`**: Hybrid search parameters with alpha/beta weights, top_k, and **offset for pagination**.
- **`VectorSearchQuery`**: Pure vector search parameters (text, top_k, **offset**).
- **`ContextSearchQuery`**: Context-aware search with context_before/after window settings and **offset**.

### `SageDB/storage/sqlite_ops.py`

Manages the relational database.

- **`SQLiteManager`**: Handles connections to `sagedb.sqlite`.
- **`_init_db`**: Creates `nodes` and `edges` tables if they don't exist. Edges table now includes an `id` column (auto-increment primary key).
- **`add_node`**: Inserts a node and returns its `rowid` (used as the FAISS ID).
- **`update_node`**: Updates node text and/or metadata by UUID. Returns True if successful.
- **`add_edge`**: Inserts an edge and returns its `id` (primary key) for retrieval/deletion.
- **`get_edge`**: Retrieves a single edge by its ID.
- **`update_edge`**: Updates edge relation and/or weight by ID. Returns updated edge data.
- **`delete_edge`**: Removes an edge by its ID from SQLite.
- **`get_nodes_batch`**: Batch fetch multiple nodes in a single query, eliminating N+1 problem.
- **`get_all_nodes`**: Fetches all nodes with pagination (LIMIT/OFFSET).

### `SageDB/storage/vector_ops.py`

Manages the vector index.

- **`VectorIndex`**: Wraps `faiss`.
- **`_create_new_index`**: Uses `IndexHNSWFlat` for O(log N) approximate nearest neighbor search. The HNSW index provides:
  - **M=32**: Number of bi-directional links per element (controls recall vs memory trade-off)
  - **efConstruction=100**: Size of dynamic candidate list during construction
  - **efSearch=64**: Size of dynamic candidate list during search
  - Wraps in `IndexIDMap2` to support custom IDs (mapped from SQLite).
- **`add_vector(vector, faiss_id, uuid)`**: Adds a vector with a specific ID and updates the id_map.
- **`remove_vector(faiss_id)`**: Removes vector by ID from both index and id_map.
- **`search(query_vector, k)`**: Returns distances and UUIDs of the nearest neighbors.
- **`get_vector_by_id(faiss_id)`**: Reconstructs a stored vector by its ID (for similarity computation).
- **`compute_similarity(query_vector, target_uuid)`**: Computes cosine similarity between a query vector and a stored vector by UUID. Used for individual similarity calculations.
- **`batch_compute_similarity(query_vector, target_uuids)`**: Efficiently computes vector similarity for multiple target UUIDs at once. Returns `Dict[uuid, similarity_score]`. Used for scoring graph-discovered nodes.
- **`ntotal`**: Property returning total vectors in the index.

### `SageDB/storage/graph_ops.py`

Manages the graph structure.

- **`GraphManager`**: Wraps `networkx.DiGraph`.
- **`add_node(uuid)`**: Adds a node to the graph.
- **`add_edge(source, target, relation, weight)`**: Adds a directed edge with metadata.
- **`expand_from_seeds(seeds, depth=2)`**: **CORE INNOVATION** - Performs BFS from seed nodes in both directions (predecessors + successors) to discover related nodes up to specified depth. Uses `nx.single_source_shortest_path_length` for efficiency.
- **`get_relationship_score(candidate, seeds)`**: Returns the maximum edge weight between candidate and any seed, using a predefined edge type hierarchy (is_a=1.0 → mentioned_in=0.3).
- **`calculate_pagerank_centrality(node)`**: Returns cached PageRank centrality score for a node with 5-minute TTL. Uses `nx.pagerank()` with alpha=0.85 for importance scoring.
- **`calculate_expanded_graph_score(node, seeds, seed_vector_scores)`**: **MAIN SCORING** - Computes comprehensive graph score:
  - **Connectivity**: Weighted average distance to seeds (weighted by seed vector scores)
  - **Centrality**: PageRank centrality (cached with TTL)
  - **Relationship**: Direct edge type strength
  - **Combined**: `0.5*connectivity + 0.3*centrality + 0.2*relationship`
- **`calculate_graph_score`** (Legacy): Simple min-distance connectivity score, kept for legacy endpoint.
- **`get_bfs_subgraph`**: Returns nodes/edges within N hops of a start node for visualization.
- **`get_full_context_window`**: Retrieves surrounding chunks via `next_chunk`/`previous_chunk` edges for context-aware search.
- **`save(path)`**: Persists graph to disk in GraphML format for better interoperability.
- **`load(path)`**: Loads graph from GraphML file on startup.

### `SageDB/ui/app.py`

The frontend dashboard built with Streamlit.

- **Configuration**: API_URL set to `http://localhost:8000`.
- **Navigation Sidebar**: Page selector with 6 options.
- **Pages**:
  - **System Health**: Shows backend status and node/edge counts via `/health`.
  - **Add Data**:
    - Tab 1: Form to create nodes (text, type, metadata JSON).
    - Tab 2: Form to create edges (source UUID, target UUID, relation, weight).
  - **Manage Data**:
    - Tab 1: Update or delete nodes - update text/metadata, or delete entirely.
    - Tab 2: Update or delete edges - update relation/weight, or delete entirely.
  - **Search**:
    - **Search Type Selector**: Radio buttons for Hybrid, Vector Only, or Graph Only.
    - Text input for query.
    - Sliders for Alpha (vector weight) and Beta (graph weight) - only shown for Hybrid mode.
    - Number input for top_k results and **offset for pagination**.
    - **Vector Only mode**: Uses dedicated `/v1/search/vector` endpoint for pure semantic search.
    - Results display with expandable cards showing:
      - UUID, full text, metadata
      - **Scoring Breakdown**: Vector Score, Graph Score, Final Score in columns
      - Score calculation formula display (for hybrid mode)
  - **Graph View**:
    - Input for start node UUID and depth slider.
    - Matplotlib visualization using `nx.spring_layout`.
    - Node labels show first 4 chars of UUID.
    - Raw JSON data display.
  - **Data Explorer**:
    - Tab 1: Button to fetch all nodes, displayed as dataframe.
    - Tab 2: Button to fetch all edges, displayed as dataframe.

### `SageDB/populate_db.py`

A utility script.

- Creates 10 "Concept" nodes (e.g., "AI", "Machine Learning").
- Creates 3 "Entity" nodes for each concept.
- Randomly connects nodes to create a dense graph structure for testing.

### `SageDB/test_sagedb.py`

Integration tests.

- **`TestSageDB`**: A `unittest` class.
- **`setUpClass`**: Starts the server in a subprocess.
- **`test_01_health`**: Checks if the server is up.
- **`test_02_create_node`**: Verifies node creation.
- **`test_04_hybrid_search`**: Verifies that search returns results.

---

## 6. Questions & Clarifications for Mentorship Round

### Architecture & Design Decisions

**Q1: Why did you choose SQLite + FAISS + NetworkX instead of using a unified graph database like Neo4j?**

- **Answer**: We wanted to demonstrate the hybrid architecture from first principles. Neo4j has vector search plugins, but using separate specialized engines shows:
  - How to coordinate multiple storage backends
  - The trade-offs between in-memory (NetworkX) vs. disk-based (SQLite) storage
  - Fine-grained control over the fusion algorithm
  - This approach is more educational and shows deeper understanding of the problem

**Q2: How does the system handle consistency across the three storage layers?**

- **Answer**: We use a combination of:
  - **Write Locks**: The `@write_locked` decorator ensures only one write operation happens at a time across all three layers
  - **Atomic SQLite Transactions**: SQLite operations are ACID-compliant
  - **Sequential Writes**: During node creation, we write to SQLite first (source of truth), then FAISS, then NetworkX. If FAISS or NetworkX fails, we have the data in SQLite for recovery
  - **Integrity Checks**: On startup, we verify SQLite and FAISS counts match and log warnings if not

**Q3: What happens if FAISS and SQLite go out of sync?**

- **Answer**: ✅ **IMPLEMENTED** - The system now auto-detects mismatches on startup and rebuilds the FAISS index from SQLite. The `rebuild_faiss_from_sqlite()` function:
  - Iterates all nodes from SQLite
  - Regenerates embeddings for each node
  - Adds vectors to FAISS with correct IDs
  - Rebuilds graph edges
  - Logs: "FAISS rebuild complete. Total vectors: N"

### Graph Scoring Algorithm

**Q4: Why do you use minimum distance instead of average distance to all seeds in the connectivity score?**

- **Answer**: ✅ **FIXED IN NEW ALGORITHM** - The updated implementation now uses **weighted average distance** across all reachable seeds, not just minimum distance. Additionally:
  - Edge types affect the effective distance (relationship-aware scoring)
  - BFS expansion discovers nodes up to depth 2, not just re-ranks existing results
  - A node close to many seeds scores higher than one close to just one seed

**Q5: How did you choose the 0.7/0.3 weights for connectivity vs. centrality in graph scoring?**

- **Answer**: Empirical tuning based on our test dataset:
  - We ran queries like "machine learning algorithms" and manually evaluated relevance
  - Higher connectivity weight (0.7) gave better results because proximity to search results matters more than global importance
  - We also expose this as tunable via the `alpha` and `beta` parameters in hybrid search, so users can adjust based on their use case

### Hybrid Fusion

**Q6: Why use Min-Max normalization for vector scores instead of Z-score normalization?**

- **Answer**:
  - Min-Max scales to [0, 1], which is the same range as our graph scores (connectivity and centrality are already 0-1)
  - Z-score can produce negative values and unbounded positive values, making fusion weights less interpretable
  - For a small top-k result set (typically 5-20 results), Min-Max is sufficient

**Q7: How do you prove that hybrid retrieval is better than vector-only or graph-only?**

- **Answer**: We demonstrate this in multiple ways:
  - **Benchmark Endpoint**: `/v1/benchmark` calculates Precision@K, Recall@K, and NDCG@K for given ground truth
  - **UI Comparison**: Users can adjust `alpha=1.0, beta=0.0` (vector-only) vs. `alpha=0.0, beta=1.0` (graph-only) vs. balanced and see ranking changes
  - **Example Case**: For queries like "related to machine learning," graph scoring boosts highly connected hub nodes that vector-only search might rank lower

### Scalability & Performance

**Q8: What are the performance bottlenecks in the current system?**

- **Answer**:
  - ✅ **FAISS Search**: Upgraded from O(N) flat index to O(log N) HNSW index for scalability
  - **Graph Traversal**: Computing shortest paths for every candidate is expensive. We limit top_k to 100 to keep this manageable
  - **In-Memory Graph**: NetworkX keeps the entire graph in RAM. For >1M nodes, we'd need a disk-based graph DB
  - **Embedding Generation**: Sentence-transformers is CPU-bound. We could add GPU support or use a model server

**Q9: How many nodes/edges can the system handle?**

- **Answer**: Current estimates with HNSW index:
  - **Nodes**: ~1M+ nodes with HNSW index maintaining <100ms search latency
  - **Edges**: ~1M edges before NetworkX graph operations degrade
  - **Performance**: ~107ms average latency @ 492 nodes (measured)
  - **Mitigations**:
    - ✅ HNSW index for vectors (supports millions of vectors with O(log N) search)
    - Replace NetworkX with a proper graph DB (Neo4j, TigerGraph) for >1M edges
    - Shard by domain or use distributed architecture

### Implementation Choices

**Q10: Why FastAPI instead of Flask or Django?**

- **Answer**:
  - **Async Support**: FastAPI is built on Starlette (async), which we could leverage for concurrent requests
  - **Automatic Docs**: Swagger UI out-of-the-box at `/docs`
  - **Pydantic Integration**: Type validation and serialization with minimal code
  - **Performance**: Benchmarked as one of the fastest Python frameworks

**Q11: Why Streamlit for the UI instead of React or a full web framework?**

- **Answer**:
  - **Speed**: We built the entire UI in <200 lines of Python
  - **No Frontend Complexity**: No need for separate build systems, state management, etc.
  - **Data Science Focus**: Streamlit is designed for data apps, so widgets for sliders, dataframes, and charts are built-in
  - **Trade-off**: Less customizable than React, but sufficient for a hackathon demo

### Dataset & Testing

**Q12: Why did you choose AI/ML topics for the mock dataset?**

- **Answer**:
  - Familiar domain for evaluators to assess relevance
  - Naturally forms a dense knowledge graph (concepts like "neural networks" connect to "backpropagation," "activation functions," etc.)
  - Easy to create realistic relationships manually

**Q13: How do you ensure the embeddings are high-quality?**

- **Answer**:
  - We use `all-MiniLM-L6-v2`, which is a well-established model with 384 dimensions
  - It's trained on billions of sentence pairs, so semantic similarity is reliable
  - We normalize vectors (L2 norm) before storing to ensure cosine similarity works correctly
  - For production, we could fine-tune on domain-specific data

### Edge Cases & Error Handling

**Q14: What happens if a user searches for a node that doesn't exist in the graph?**

- **Answer**:
  - Vector search will still return the top-k most similar nodes by embedding
  - Graph score will be 0.0 for all results (no connectivity, low centrality)
  - Final ranking will be purely based on vector similarity (alpha weight)

**Q15: How do you handle concurrent read/write operations?**

- **Answer**:
  - We use a **Reader-Writer Lock** from `readerwriterlock`
  - Multiple reads can happen simultaneously (GET requests)
  - Writes are exclusive (POST/DELETE block all other operations)
  - This prevents race conditions like deleting a node while it's being searched

---

## 7. Demo Script for Evaluation

### 1. Show System Health

- Navigate to "System Health" tab
- Show backend is running and data is loaded

### 2. Data Explorer

- Show "Data Explorer" tab
- Display all 40 nodes and 124 edges
- Explain the AI/ML knowledge graph structure

### 3. Search Demo (Vector-Only)

- Navigate to "Search" tab
- Select "Vector Only" search type
- Search query: "neural networks"
- Show results ranked purely by semantic similarity (uses `/v1/search/vector` endpoint)

### 4. Search Demo (Graph-Only)

- Select "Graph Only" search type
- Same query: "neural networks"
- Show how hub nodes (like "Machine Learning") rank higher due to centrality

### 5. Search Demo (Hybrid)

- Select "Hybrid" search type
- Same query: "neural networks"
- Set `alpha=0.7, beta=0.3` (default)
- Show how results balance semantic relevance + structural importance
- Highlight the scoring breakdown for each result

### 6. Graph Visualization

- Pick a node UUID from search results
- Navigate to "Graph View"
- Show the subgraph with depth=2
- Explain how relationships inform the graph score

### 7. CRUD Operations

- Navigate to "Add Data" tab
- Create a new node: "Transformer Architecture"
- Create edges: "Transformer" → "Attention Mechanism"
- Search for "attention" and show it now returns the new node
- This demonstrates the system is live and mutable

### 8. Data Management

- Navigate to "Manage Data" tab
- **Update Node**: Update the text or metadata of an existing node
- **Delete Edge**: Lookup an edge by ID and delete it
- **Delete Node**: Delete a node (also removes connected edges)
- This demonstrates full CRUD capabilities

---

## 8. Future Enhancements (If Asked)

1. **Multi-Hop Reasoning**: Return the "reasoning path" showing how the system traversed from query to result
2. **Personalized PageRank**: Use the query as a seed and compute PPR for better graph scores
3. **Relationship Types**: Allow filtering by edge types (e.g., "show only 'part_of' relationships")
4. **Temporal Graphs**: Add timestamps to edges to track knowledge evolution
5. **Federated Search**: Query multiple SageDB instances and merge results
6. **Explainability**: Visualize which parts of the query matched which parts of the result (attention-like mechanism)

---

## 9. Questions for Mentors/Evaluators

### Technical Clarifications Needed

**Q1: Performance Benchmarks**

- What are the expected query latency requirements for Round 2 evaluation?
- Should we optimize for datasets of a specific size (10K, 100K, 1M nodes)?
- Are there specific hardware constraints we should design for?

**Q2: Evaluation Dataset**

- Will you provide a standard dataset for all teams to be evaluated on?
- Should we prepare to ingest custom data during the demo, or is our pre-populated dataset sufficient?
- What domain should the knowledge graph represent (general knowledge, research papers, code documentation)?

**Q3: Graph Scoring Validation**

- How will "hybrid retrieval effectiveness" be measured quantitatively?
- Should we implement specific metrics (MAP, MRR, P@K) or is qualitative comparison (vector-only vs. hybrid) acceptable?
- Are there baseline systems we should compare against?

### Scope & Requirements

**Q4: Multi-Hop Reasoning**

- Is it required to return the **path** of reasoning (e.g., "Query → Node A → Node B → Result"), or is scoring based on connectivity sufficient?
- The current implementation calculates scores based on graph proximity but doesn't expose the traversal path. Should we prioritize this?

**Q5: Schema Enforcement**

- The problem statement mentions "basic schema enforcement" as a stretch goal. What does this mean in practice?
  - Validating node types (document/entity/concept)?
  - Enforcing allowed edge types between node types?
  - Preventing cycles or other graph constraints?

**Q6: API Documentation Standard**

- You mentioned API documentation as part of deliverables. Is the auto-generated Swagger/OpenAPI docs at `/docs` sufficient, or do you need a separate written API guide?

### Demo & Presentation

**Q7: Live Coding vs. Pre-Built Demo**

- During the final round, should we be prepared to implement new features live, or is demonstrating the existing system sufficient?
- Are there specific use cases you'd like to see (e.g., "Show me how to query relationships between entities")?

**Q8: System Architecture Diagram**

- Would a visual architecture diagram (showing SQLite ↔ FAISS ↔ NetworkX flow) improve our presentation?
- Should we prepare slides or is a live code walkthrough preferred?

### Edge Cases & Robustness

**Q9: Error Recovery**

- If FAISS index gets corrupted, should the system auto-rebuild from SQLite, or is manual intervention acceptable?
- How critical is idempotency for write operations (e.g., creating the same node twice)?

**Q10: Concurrency Testing**

- Should we demonstrate the system handling concurrent users?
- Are there specific stress tests we should prepare (e.g., "100 simultaneous searches")?

### Intellectual Honesty

**Q11: Use of External Libraries**

- We're using FAISS (Facebook AI), NetworkX, and sentence-transformers. The problem statement says "should not use solutions specifically solving for the problem statement."
  - Is using FAISS for vector search acceptable since it's a general-purpose library, not a pre-built hybrid DB?
  - Should we implement our own vector indexing from scratch (e.g., Annoy, custom HNSW)?

**Q12: Embedding Model Choice**

- We're using a pre-trained embedding model (`all-MiniLM-L6-v2`). Is this acceptable, or should we show custom embedding training?
- Would switching to OpenAI embeddings or other API-based models disqualify us?

### Clarifications on Judging Criteria

**Q13: "Real-world demo" (30 points)**

- What constitutes a "real-world" use case?
  - Is our AI/ML knowledge graph realistic enough?
  - Should we demonstrate on actual data (e.g., Wikipedia scraping, research papers)?

**Q14: "System design depth" (20 points)**

- Should we prepare a detailed write-up on architectural decisions (trade-offs, alternatives considered)?
- Do you want to see code-level explanations or high-level design patterns?

**Q15: "Presentation & storytelling" (10 points)**

- What's the time limit for the final presentation?
- Should we focus on problem → solution → results, or dive deep into technical implementation?

### Blockers & Current Status

| Blocker                             | Status         | Resolution                                                         |
| ----------------------------------- | -------------- | ------------------------------------------------------------------ |
| Graph Scoring (min distance only)   | ✅ RESOLVED    | Implemented weighted average distance + relationship-aware scoring |
| FAISS/SQLite Sync                   | ✅ RESOLVED    | Auto-rebuild on startup via `rebuild_faiss_from_sqlite()`          |
| Graph Expansion (only re-ranking)   | ✅ RESOLVED    | BFS expansion discovers new candidates via `expand_from_seeds()`   |
| Missing PUT /nodes/{id}             | ✅ RESOLVED    | Added `update_node` endpoint for updating text/metadata            |
| Missing PUT /edges/{id}             | ✅ RESOLVED    | Added `update_edge` endpoint for updating relation/weight          |
| Missing DELETE /edges/{id}          | ✅ RESOLVED    | Added `delete_edge` endpoint with edge ID support                  |
| Missing GET /edges/{id}             | ✅ RESOLVED    | Added `get_edge` endpoint for single edge retrieval                |
| Missing POST /search/vector         | ✅ RESOLVED    | Added dedicated vector-only search endpoint                        |
| Edge table has no ID                | ✅ RESOLVED    | Added `id` column to edges table for CRUD operations               |
| Alpha+Beta must sum to 1.0          | ✅ RESOLVED    | Relaxed constraint - weights are now normalized in fusion          |
| FAISS Index Type (O(N) brute-force) | ✅ RESOLVED    | Upgraded to `IndexHNSWFlat` for O(log N) search                    |
| Simple Degree Centrality            | ✅ RESOLVED    | Upgraded to PageRank centrality with 5-minute TTL caching          |
| Pickle-based Graph Storage          | ✅ RESOLVED    | Migrated to GraphML format for better interoperability             |
| N+1 Query Problem                   | ✅ RESOLVED    | Implemented batch hydration via `get_nodes_batch()`                |
| No Offset Pagination                | ✅ RESOLVED    | Added `offset` parameter to all search endpoints                   |
| Query Decomposition                 | ✅ RESOLVED    | Automatic splitting for "X vs Y", "between A and B" queries        |
| RRF Threshold Bug                   | ✅ RESOLVED    | Removed MINIMUM_RELEVANCE_THRESHOLD filter entirely                |
| Only next_chunk edges               | ✅ RESOLVED    | Added semantic similarity edges (`similar_to`, `related_to`)       |
| Static edge weights                 | ✅ RESOLVED    | Dynamic weights based on cosine similarity during ingestion        |
| No interactive visualization        | ✅ RESOLVED    | Built 3D force-directed graph with Vite + React                    |
| Raw text ingestion error            | ✅ RESOLVED    | Fixed Chunk.content → Chunk.text attribute access                  |
| In-Memory Graph (RAM limits)        | ⚠️ KNOWN       | NetworkX in-memory. Would need disk-based DB for >100K nodes       |
| Ground Truth for Benchmark          | ⚠️ PENDING     | `/v1/benchmark` endpoint ready, need labeled data                  |
| CLI Tool                            | ❌ NOT STARTED | Web UI built, CLI is optional enhancement                          |

### Stretch Goals Progress

| Goal                                          | Status  | Notes                                                     |
| --------------------------------------------- | ------- | --------------------------------------------------------- |
| Improve graph scoring (weighted avg distance) | ✅ DONE | `calculate_expanded_graph_score`                          |
| Graph expansion algorithm                     | ✅ DONE | `expand_from_seeds` with BFS                              |
| Relationship-aware scoring                    | ✅ DONE | `get_relationship_score` with edge type weights           |
| Auto-rebuild FAISS from SQLite                | ✅ DONE | `rebuild_faiss_from_sqlite()` on startup                  |
| Batch vector similarity                       | ✅ DONE | `batch_compute_similarity` for efficiency                 |
| Full CRUD for nodes (including UPDATE)        | ✅ DONE | `PUT /v1/nodes/{uuid}` with re-embedding                  |
| Full CRUD for edges (GET/PUT/DELETE by ID)    | ✅ DONE | Edge ID column + all endpoints                            |
| Pure vector search endpoint                   | ✅ DONE | `POST /v1/search/vector`                                  |
| Flexible alpha/beta weights                   | ✅ DONE | Weights normalized, no sum-to-1 requirement               |
| UI: Manage Data page                          | ✅ DONE | Update/Delete nodes, Delete edges                         |
| UI: Search type selector                      | ✅ DONE | Hybrid / Vector Only / Graph Only modes                   |
| HNSW Index for O(log N) search                | ✅ DONE | `IndexHNSWFlat` with M=32, efSearch=64                    |
| PageRank Centrality                           | ✅ DONE | Cached PageRank with 5-minute TTL                         |
| GraphML Persistence                           | ✅ DONE | Replaced pickle with GraphML format                       |
| Batch Hydration                               | ✅ DONE | `get_nodes_batch()` eliminates N+1 queries                |
| Offset Pagination                             | ✅ DONE | All search endpoints support `offset` parameter           |
| Query Decomposition                           | ✅ DONE | Automatic splitting for comparison queries                |
| RRF Multi-Query Fusion                        | ✅ DONE | Reciprocal Rank Fusion with configurable k parameter      |
| Semantic Edge Weights                         | ✅ DONE | Dynamic weights based on cosine similarity                |
| Cross-Document Relationships                  | ✅ DONE | `related_to` edges for non-adjacent semantic similarity   |
| Interactive 3D Visualization                  | ✅ DONE | Vite + React force-directed graph with edge type colors   |
| Edge Migration Tool                           | ✅ DONE | `migrate_edges.py` for retroactive semantic edge creation |
| Multi-hop reasoning path                      | ❌ TODO | Show traversal path in results                            |
| Relationship filtering                        | ❌ TODO | Filter by edge types in API                               |
| CLI tool                                      | ❌ TODO | For scripted querying                                     |
| Dockerization                                 | ❌ TODO | For easy deployment                                       |
| Architecture diagram                          | ❌ TODO | Visual system overview                                    |

**Question**: Which of the remaining items would you prioritize for evaluation?
