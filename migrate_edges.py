"""
IMPORTANT: Run this migration with the API server STOPPED!

This script adds semantic similarity edges to existing chunks retroactively.
"""

import sqlite3
import numpy as np
import logging
from tqdm import tqdm

from core.embedding import embedding_service
from ingestion.config import EDGE_WEIGHTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate():
    """Add semantic similarity edges to existing chunks."""
    
    conn = sqlite3.connect('data/sqlite.db', timeout=30.0)
    cursor = conn.cursor()
    
    # Get all chunks
    logger.info("Fetching all chunks...")
    chunks = cursor.execute("""
        SELECT n.uuid, n.text, c.document_id, c.chunk_index
        FROM chunks c
        JOIN nodes n ON c.node_uuid = n.uuid
        ORDER BY c.document_id, c.chunk_index
    """).fetchall()
    
    if not chunks:
        logger.warning("No chunks found!")
        return
    
    logger.info(f"Found {len(chunks)} chunks in {len(set(c[2] for c in chunks))} documents")
    
    # Extract data
    chunk_uuids = [c[0] for c in chunks]
    chunk_texts = [c[1] for c in chunks]
    chunk_doc_ids = [c[2] for c in chunks]
    chunk_positions = [c[3] for c in chunks]
    
    # Compute embeddings
    logger.info("Computing embeddings...")
    embeddings = embedding_service.encode_batch(chunk_texts, batch_size=32)
    
    edges_added = 0
    edges_updated = 0
    
    # Group chunks by document
    doc_chunks = {}
    for i, doc_id in enumerate(chunk_doc_ids):
        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = []
        doc_chunks[doc_id].append(i)
    
    logger.info(f"Processing {len(doc_chunks)} documents...")
    
    for doc_id, indices in tqdm(doc_chunks.items(), desc="Documents"):
        indices = sorted(indices, key=lambda i: chunk_positions[i])
        
        # 1. Update sequential edge weights with similarity boost
        for i in range(len(indices) - 1):
            idx_curr = indices[i]
            idx_next = indices[i + 1]
            
            cosine_sim = float(np.dot(embeddings[idx_curr], embeddings[idx_next]))
            
            # Dynamic weight calculation
            base_weight = EDGE_WEIGHTS["next_chunk"]
            similarity_boost = 0.15 * cosine_sim
            dynamic_weight = min(1.0, base_weight * (1 + similarity_boost))
            
            # Update existing next_chunk edge
            cursor.execute("""
                UPDATE edges 
                SET weight = ?
                WHERE source_uuid = ? AND target_uuid = ? AND relation = 'next_chunk'
            """, (dynamic_weight, chunk_uuids[idx_curr], chunk_uuids[idx_next]))
            
            if cursor.rowcount > 0:
                edges_updated += 1
            
            # Add similar_to edge if very similar
            if cosine_sim > 0.80:
                sim_weight = EDGE_WEIGHTS["similar_to"] * cosine_sim
                try:
                    cursor.execute("""
                        INSERT INTO edges (source_uuid, target_uuid, relation, weight)
                        VALUES (?, ?, ?, ?)
                    """, (chunk_uuids[idx_curr], chunk_uuids[idx_next], "similar_to", sim_weight))
                    edges_added += 1
                except sqlite3.IntegrityError:
                    pass  # Edge already exists
        
        # 2. Add cross-chunk semantic similarity edges
        SIMILARITY_THRESHOLD = 0.75
        
        for i in range(len(indices)):
            for j in range(i + 2, len(indices)):  # Skip adjacent
                idx_i = indices[i]
                idx_j = indices[j]
                
                cosine_sim = float(np.dot(embeddings[idx_i], embeddings[idx_j]))
                
                if cosine_sim >= SIMILARITY_THRESHOLD:
                    weight = EDGE_WEIGHTS["related_to"] * cosine_sim
                    
                    # Bidirectional edges
                    for src, tgt in [(idx_i, idx_j), (idx_j, idx_i)]:
                        try:
                            cursor.execute("""
                                INSERT INTO edges (source_uuid, target_uuid, relation, weight)
                                VALUES (?, ?, ?, ?)
                            """, (chunk_uuids[src], chunk_uuids[tgt], "related_to", weight))
                            edges_added += 1
                        except sqlite3.IntegrityError:
                            pass
    
    conn.commit()
    conn.close()
    
    logger.info(f"✅ Migration complete!")
    logger.info(f"   - Updated {edges_updated} next_chunk edge weights")
    logger.info(f"   - Added {edges_added} new semantic edges (similar_to + related_to)")
    
    # Show final stats
    conn2 = sqlite3.connect('data/sqlite.db')
    stats = conn2.execute("SELECT relation, COUNT(*) FROM edges GROUP BY relation").fetchall()
    logger.info("\n📊 Final edge statistics:")
    for relation, count in stats:
        logger.info(f"   {relation}: {count}")
    conn2.close()

if __name__ == "__main__":
    print("=" * 60)
    print("SEMANTIC EDGE MIGRATION")
    print("=" * 60)
    print("\n⚠️  IMPORTANT: Make sure the API server is STOPPED before running this!\n")
    input("Press Enter to continue or Ctrl+C to cancel...")
    
    logger.info("Starting migration...")
    migrate()
    
    print("\n" + "=" * 60)
    print("✅ Migration complete! You can now restart the server.")
    print("=" * 60)
