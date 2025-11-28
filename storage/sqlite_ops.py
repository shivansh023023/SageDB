import sqlite3
import json
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from config import SQLITE_DB_PATH

logger = logging.getLogger(__name__)

class SQLiteManager:
    def __init__(self, db_path: str = SQLITE_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize the database schema."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Nodes Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                uuid TEXT PRIMARY KEY,
                faiss_id INTEGER UNIQUE NOT NULL,
                text TEXT NOT NULL,
                metadata JSON,
                type TEXT CHECK(type IN ('document', 'entity', 'concept'))
            );
        """)
        
        # Check if edges table exists and needs migration
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='edges';")
        edges_table_exists = cursor.fetchone() is not None
        
        if edges_table_exists:
            # Check if 'id' column exists
            cursor.execute("PRAGMA table_info(edges);")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'id' not in columns:
                logger.info("Migrating edges table to add 'id' column...")
                # Disable foreign keys during migration
                cursor.execute("PRAGMA foreign_keys = OFF;")
                
                # Drop any leftover temp table from failed migration
                cursor.execute("DROP TABLE IF EXISTS edges_new;")
                
                # Create new table with ID column
                cursor.execute("""
                    CREATE TABLE edges_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_uuid TEXT,
                        target_uuid TEXT,
                        relation TEXT NOT NULL,
                        weight REAL CHECK(weight > 0 AND weight <= 1.0),
                        UNIQUE(source_uuid, target_uuid, relation)
                    );
                """)
                cursor.execute("""
                    INSERT INTO edges_new (source_uuid, target_uuid, relation, weight)
                    SELECT source_uuid, target_uuid, relation, weight FROM edges;
                """)
                cursor.execute("DROP TABLE edges;")
                cursor.execute("ALTER TABLE edges_new RENAME TO edges;")
                
                # Re-enable foreign keys
                cursor.execute("PRAGMA foreign_keys = ON;")
                logger.info("Edges table migration complete.")
        else:
            # Create new edges table with ID
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_uuid TEXT,
                    target_uuid TEXT,
                    relation TEXT NOT NULL,
                    weight REAL CHECK(weight > 0 AND weight <= 1.0),
                    UNIQUE(source_uuid, target_uuid, relation)
                );
            """)
        
        # Enable foreign keys for normal operations
        cursor.execute("PRAGMA foreign_keys = ON;")
        
        # FAISS Metadata Table (for atomic counter)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faiss_metadata (
                key TEXT PRIMARY KEY,
                value INTEGER
            );
        """)
        
        # Initialize next_id if not exists
        cursor.execute("INSERT OR IGNORE INTO faiss_metadata (key, value) VALUES ('next_id', 0);")
        
        # Documents Table (for ingestion pipeline - stores source documents)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                content_hash TEXT UNIQUE NOT NULL,
                original_size INTEGER,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON
            );
        """)
        
        # Chunks Table (for ingestion pipeline - links chunks to source documents)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                node_uuid TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                section_hierarchy TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY (node_uuid) REFERENCES nodes(uuid) ON DELETE CASCADE,
                UNIQUE(document_id, chunk_index)
            );
        """)
        
        # Create index for faster lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_node ON chunks(node_uuid);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);")
        
        conn.commit()
        conn.close()

    def add_node(self, uuid: str, text: str, node_type: str, metadata: Dict) -> int:
        """
        Atomically allocate FAISS ID and insert node.
        Returns the allocated faiss_id.
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            # Atomic ID allocation
            cursor.execute("UPDATE faiss_metadata SET value = value + 1 WHERE key='next_id' RETURNING value")
            next_val = cursor.fetchone()[0]
            faiss_id = next_val - 1
            
            cursor.execute(
                "INSERT INTO nodes (uuid, faiss_id, text, type, metadata) VALUES (?, ?, ?, ?, ?)",
                (uuid, faiss_id, text, node_type, json.dumps(metadata))
            )
            conn.commit()
            return faiss_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add node to SQLite: {e}")
            raise e
        finally:
            conn.close()

    def get_node(self, uuid: str) -> Optional[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT uuid, faiss_id, text, type, metadata FROM nodes WHERE uuid = ?", (uuid,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "uuid": row[0],
                "faiss_id": row[1],
                "text": row[2],
                "type": row[3],
                "metadata": json.loads(row[4]) if row[4] else {}
            }
        return None

    def get_nodes_batch(self, uuids: List[str]) -> Dict[str, Dict]:
        """
        Batch fetch multiple nodes in a single query.
        
        This eliminates the N+1 query problem where fetching N nodes
        requires N separate database round-trips.
        
        Args:
            uuids: List of node UUIDs to fetch
            
        Returns:
            Dict mapping UUID -> node data
        """
        if not uuids:
            return {}
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Use parameterized IN clause for safety
        placeholders = ','.join(['?'] * len(uuids))
        query = f"SELECT uuid, faiss_id, text, type, metadata FROM nodes WHERE uuid IN ({placeholders})"
        
        cursor.execute(query, uuids)
        rows = cursor.fetchall()
        conn.close()
        
        results = {}
        for row in rows:
            results[row[0]] = {
                "uuid": row[0],
                "faiss_id": row[1],
                "text": row[2],
                "type": row[3],
                "metadata": json.loads(row[4]) if row[4] else {}
            }
        return results

    def get_filtered_node_ids(
        self, 
        metadata_filter: Optional[Dict[str, str]] = None, 
        node_type_filter: Optional[str] = None,
        limit: int = 10000
    ) -> List[int]:
        """
        Pre-filter nodes by metadata and/or type BEFORE vector search.
        
        This enables efficient filtered search by:
        1. Filtering in SQLite (fast, indexed)
        2. Returning only FAISS IDs that match
        3. Vector search then only considers these IDs
        
        Args:
            metadata_filter: Dict of key-value pairs to match in metadata JSON
            node_type_filter: Filter by node type (document, entity, concept)
            limit: Maximum number of IDs to return
            
        Returns:
            List of FAISS IDs matching the filter criteria
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Build dynamic WHERE clause
        conditions = []
        params = []
        
        if node_type_filter:
            conditions.append("type = ?")
            params.append(node_type_filter)
        
        if metadata_filter:
            for key, value in metadata_filter.items():
                # Use JSON extraction for metadata filtering
                conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                params.append(value)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT faiss_id FROM nodes WHERE {where_clause} LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [row[0] for row in rows]

    def get_chunk_provenance(self, node_uuid: str) -> Optional[Dict]:
        """
        Get chunking provenance information for a node.
        
        Returns source document info, chunk index, and section hierarchy
        for enhanced visibility into where content came from.
        
        Args:
            node_uuid: UUID of the node to get provenance for
            
        Returns:
            Dict with source_document, chunk_index, total_chunks, section_path, etc.
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Join chunks with documents to get full provenance
        cursor.execute("""
            SELECT 
                d.filename as source_document,
                c.chunk_index,
                (SELECT COUNT(*) FROM chunks WHERE document_id = c.document_id) as total_chunks,
                c.section_hierarchy,
                d.file_type,
                d.ingested_at
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.node_uuid = ?
        """, (node_uuid,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            section_path = json.loads(row[3]) if row[3] else None
            return {
                "source_document": row[0],
                "chunk_index": row[1],
                "total_chunks": row[2],
                "section_path": section_path,
                "file_type": row[4],
                "ingested_at": row[5]
            }
        return None

    def get_chunks_provenance_batch(self, node_uuids: List[str]) -> Dict[str, Dict]:
        """
        Batch fetch provenance for multiple nodes.
        
        Args:
            node_uuids: List of node UUIDs
            
        Returns:
            Dict mapping UUID -> provenance data
        """
        if not node_uuids:
            return {}
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        placeholders = ','.join(['?'] * len(node_uuids))
        cursor.execute(f"""
            SELECT 
                c.node_uuid,
                d.filename as source_document,
                c.chunk_index,
                (SELECT COUNT(*) FROM chunks c2 WHERE c2.document_id = c.document_id) as total_chunks,
                c.section_hierarchy,
                d.file_type
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.node_uuid IN ({placeholders})
        """, node_uuids)
        
        rows = cursor.fetchall()
        conn.close()
        
        results = {}
        for row in rows:
            section_path = json.loads(row[4]) if row[4] else None
            results[row[0]] = {
                "source_document": row[1],
                "chunk_index": row[2],
                "total_chunks": row[3],
                "section_path": section_path,
                "file_type": row[5]
            }
        return results

    def delete_node(self, uuid: str) -> Optional[int]:
        """Deletes node and returns its faiss_id if it existed."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT faiss_id FROM nodes WHERE uuid = ?", (uuid,))
            row = cursor.fetchone()
            if not row:
                return None
            
            faiss_id = row[0]
            cursor.execute("DELETE FROM nodes WHERE uuid = ?", (uuid,))
            conn.commit()
            return faiss_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete node from SQLite: {e}")
            raise e
        finally:
            conn.close()

    def add_edge(self, source_uuid: str, target_uuid: str, relation: str, weight: float) -> int:
        """Add edge and return its ID."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO edges (source_uuid, target_uuid, relation, weight) VALUES (?, ?, ?, ?) RETURNING id",
                (source_uuid, target_uuid, relation, weight)
            )
            edge_id = cursor.fetchone()[0]
            conn.commit()
            return edge_id
        except sqlite3.IntegrityError as e:
            conn.rollback()
            if "UNIQUE constraint" in str(e):
                raise ValueError(f"Edge already exists between these nodes with this relation")
            raise ValueError(f"Source or target node does not exist: {e}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add edge to SQLite: {e}")
            raise e
        finally:
            conn.close()

    def add_edge_safe(self, source_uuid: str, target_uuid: str, relation: str, weight: float) -> Optional[int]:
        """Add edge, ignoring duplicates. Returns edge ID or None if duplicate."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO edges (source_uuid, target_uuid, relation, weight) VALUES (?, ?, ?, ?)",
                (source_uuid, target_uuid, relation, weight)
            )
            conn.commit()
            if cursor.rowcount > 0:
                # Get the last inserted ID
                cursor.execute("SELECT last_insert_rowid()")
                return cursor.fetchone()[0]
            return None  # Duplicate was ignored
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add edge to SQLite: {e}")
            return None
        finally:
            conn.close()

    def get_edge(self, edge_id: int) -> Optional[Dict]:
        """Get edge by ID."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT id, source_uuid, target_uuid, relation, weight FROM edges WHERE id = ?", (edge_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row[0],
                "source_id": row[1],
                "target_id": row[2],
                "relation": row[3],
                "weight": row[4]
            }
        return None

    def delete_edge(self, edge_id: int) -> bool:
        """Delete edge by ID. Returns True if deleted, False if not found."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete edge from SQLite: {e}")
            raise e
        finally:
            conn.close()

    def update_edge(self, edge_id: int, relation: Optional[str] = None, weight: Optional[float] = None) -> Optional[Dict]:
        """Update edge relation and/or weight. Returns updated edge if found, None otherwise."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            # Check if edge exists
            cursor.execute("SELECT id FROM edges WHERE id = ?", (edge_id,))
            if not cursor.fetchone():
                return None
            
            if relation is not None and weight is not None:
                cursor.execute("UPDATE edges SET relation = ?, weight = ? WHERE id = ?", 
                              (relation, weight, edge_id))
            elif relation is not None:
                cursor.execute("UPDATE edges SET relation = ? WHERE id = ?", (relation, edge_id))
            elif weight is not None:
                cursor.execute("UPDATE edges SET weight = ? WHERE id = ?", (weight, edge_id))
            
            conn.commit()
            
            # Return updated edge
            return self.get_edge(edge_id)
        except sqlite3.IntegrityError as e:
            conn.rollback()
            if "UNIQUE constraint" in str(e):
                raise ValueError(f"Edge with this relation already exists between these nodes")
            raise e
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update edge in SQLite: {e}")
            raise e
        finally:
            conn.close()

    def update_node(self, uuid: str, text: Optional[str] = None, metadata: Optional[Dict] = None) -> bool:
        """Update node text and/or metadata. Returns True if updated, False if not found."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            # Check if node exists
            cursor.execute("SELECT uuid FROM nodes WHERE uuid = ?", (uuid,))
            if not cursor.fetchone():
                return False
            
            if text is not None and metadata is not None:
                cursor.execute("UPDATE nodes SET text = ?, metadata = ? WHERE uuid = ?", 
                              (text, json.dumps(metadata), uuid))
            elif text is not None:
                cursor.execute("UPDATE nodes SET text = ? WHERE uuid = ?", (text, uuid))
            elif metadata is not None:
                cursor.execute("UPDATE nodes SET metadata = ? WHERE uuid = ?", (json.dumps(metadata), uuid))
            
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update node in SQLite: {e}")
            raise e
        finally:
            conn.close()

    def get_all_nodes_map(self) -> Dict[int, str]:
        """Returns a mapping of faiss_id -> uuid for rebuilding in-memory map."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT faiss_id, uuid FROM nodes")
        rows = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}

    def count_nodes(self) -> int:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM nodes")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def count_edges(self) -> int:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM edges")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_all_nodes(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT uuid, faiss_id, text, type, metadata FROM nodes LIMIT ? OFFSET ?", (limit, offset))
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            "uuid": row[0],
            "faiss_id": row[1],
            "text": row[2],
            "type": row[3],
            "metadata": json.loads(row[4]) if row[4] else {}
        } for row in rows]

    def get_all_edges(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT id, source_uuid, target_uuid, relation, weight FROM edges LIMIT ? OFFSET ?", (limit, offset))
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            "id": row[0],
            "source_id": row[1],
            "target_id": row[2],
            "relation": row[3],
            "weight": row[4]
        } for row in rows]

    # ==================== Document/Chunk Methods for Ingestion ====================
    
    def add_document(self, filename: str, file_type: str, content_hash: str,
                     original_size: int, metadata: Optional[Dict] = None) -> int:
        """Add a document record and return its ID."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """INSERT INTO documents (filename, file_type, content_hash, original_size, metadata)
                   VALUES (?, ?, ?, ?, ?) RETURNING id""",
                (filename, file_type, content_hash, original_size, 
                 json.dumps(metadata) if metadata else None)
            )
            doc_id = cursor.fetchone()[0]
            conn.commit()
            return doc_id
        except sqlite3.IntegrityError as e:
            conn.rollback()
            if "UNIQUE constraint" in str(e):
                raise ValueError(f"Document with this content hash already exists")
            raise e
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add document to SQLite: {e}")
            raise e
        finally:
            conn.close()

    def get_document_by_hash(self, content_hash: str) -> Optional[Dict]:
        """Get document by content hash."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, filename, file_type, content_hash, original_size, ingested_at, metadata FROM documents WHERE content_hash = ?",
            (content_hash,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row[0],
                "filename": row[1],
                "file_type": row[2],
                "content_hash": row[3],
                "original_size": row[4],
                "ingested_at": row[5],
                "metadata": json.loads(row[6]) if row[6] else {}
            }
        return None

    def get_document_by_id(self, doc_id: int) -> Optional[Dict]:
        """Get document by ID."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, filename, file_type, content_hash, original_size, ingested_at, metadata FROM documents WHERE id = ?",
            (doc_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row[0],
                "filename": row[1],
                "file_type": row[2],
                "content_hash": row[3],
                "original_size": row[4],
                "ingested_at": row[5],
                "metadata": json.loads(row[6]) if row[6] else {}
            }
        return None

    def add_chunk(self, document_id: int, node_uuid: str, chunk_index: int,
                  section_hierarchy: Optional[str] = None) -> int:
        """Add a chunk record linking a node to its source document."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """INSERT INTO chunks (document_id, node_uuid, chunk_index, section_hierarchy)
                   VALUES (?, ?, ?, ?) RETURNING id""",
                (document_id, node_uuid, chunk_index, section_hierarchy)
            )
            chunk_id = cursor.fetchone()[0]
            conn.commit()
            return chunk_id
        except sqlite3.IntegrityError as e:
            conn.rollback()
            if "UNIQUE constraint" in str(e):
                raise ValueError(f"Chunk with this index already exists for this document")
            raise e
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add chunk to SQLite: {e}")
            raise e
        finally:
            conn.close()

    def get_chunks_by_document(self, document_id: int) -> List[Dict]:
        """Get all chunks for a document."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT c.id, c.document_id, c.node_uuid, c.chunk_index, c.section_hierarchy, c.created_at,
                      n.text, n.metadata
               FROM chunks c
               JOIN nodes n ON c.node_uuid = n.uuid
               WHERE c.document_id = ?
               ORDER BY c.chunk_index""",
            (document_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            "id": row[0],
            "document_id": row[1],
            "node_uuid": row[2],
            "chunk_index": row[3],
            "section_hierarchy": row[4],
            "created_at": row[5],
            "text": row[6],
            "metadata": json.loads(row[7]) if row[7] else {}
        } for row in rows]

    def get_chunk_by_node(self, node_uuid: str) -> Optional[Dict]:
        """Get chunk info for a node (to trace back to source document)."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT c.id, c.document_id, c.node_uuid, c.chunk_index, c.section_hierarchy,
                      d.filename, d.file_type
               FROM chunks c
               JOIN documents d ON c.document_id = d.id
               WHERE c.node_uuid = ?""",
            (node_uuid,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "chunk_id": row[0],
                "document_id": row[1],
                "node_uuid": row[2],
                "chunk_index": row[3],
                "section_hierarchy": row[4],
                "source_filename": row[5],
                "source_file_type": row[6]
            }
        return None

    def get_all_documents(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get all documents with pagination."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, filename, file_type, content_hash, original_size, ingested_at, metadata
               FROM documents ORDER BY ingested_at DESC LIMIT ? OFFSET ?""",
            (limit, offset)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            "id": row[0],
            "filename": row[1],
            "file_type": row[2],
            "content_hash": row[3],
            "original_size": row[4],
            "ingested_at": row[5],
            "metadata": json.loads(row[6]) if row[6] else {}
        } for row in rows]

    def count_documents(self) -> int:
        """Count total documents."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document and all its chunks (cascades to chunks table)."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            # First get all node UUIDs for this document's chunks
            cursor.execute("SELECT node_uuid FROM chunks WHERE document_id = ?", (doc_id,))
            node_uuids = [row[0] for row in cursor.fetchall()]
            
            # Delete the document (will cascade to chunks)
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            
            return deleted, node_uuids  # Return UUIDs so caller can clean up nodes/vectors
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete document from SQLite: {e}")
            raise e
        finally:
            conn.close()

    def nuke_all(self) -> Dict[str, int]:
        """Delete ALL data from all tables. Returns counts of deleted items."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            # Get counts before deletion
            cursor.execute("SELECT COUNT(*) FROM nodes")
            node_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM edges")
            edge_count = cursor.fetchone()[0]
            
            # Delete all data
            cursor.execute("DELETE FROM edges")
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM documents")
            cursor.execute("DELETE FROM nodes")
            
            # Reset FAISS ID counter
            cursor.execute("UPDATE faiss_metadata SET value = 0 WHERE key = 'next_id'")
            
            conn.commit()
            logger.info(f"Nuked all data: {node_count} nodes, {edge_count} edges")
            return {"nodes": node_count, "edges": edge_count}
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to nuke database: {e}")
            raise e
        finally:
            conn.close()

sqlite_manager = SQLiteManager()
