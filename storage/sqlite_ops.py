import sqlite3
import json
import logging
from typing import Optional, Dict, List, Tuple
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

sqlite_manager = SQLiteManager()
