from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import numpy as np
from sklearn.decomposition import PCA
from storage.vector_ops import vector_index
from storage.sqlite_ops import sqlite_manager

router = APIRouter(prefix="/v1/viz", tags=["Visualization"])

@router.get("/vector-space")
async def get_vector_space(limit: int = 2000):
    """
    Get 3D coordinates of nodes based on their vector embeddings using PCA.
    """
    try:
        # 1. Fetch all nodes with embeddings
        # We need to reconstruct vectors from FAISS
        
        # Get all node UUIDs and their FAISS IDs
        id_map = vector_index.id_map
        if not id_map:
            return {"nodes": []}
            
        # Limit to avoid performance issues
        faiss_ids = list(id_map.keys())[:limit]
        
        vectors = []
        nodes = []
        
        for fid in faiss_ids:
            uuid = id_map[fid]
            vector = vector_index.get_vector_by_id(fid)
            
            if vector is not None:
                vectors.append(vector)
                
                # Get node metadata for label
                node_data = sqlite_manager.get_node(uuid)
                nodes.append({
                    "id": uuid,
                    "text": node_data['text'] if node_data else "Unknown",
                    "group": node_data['metadata'].get('type', 'unknown') if node_data and node_data.get('metadata') else 'unknown'
                })
        
        if not vectors:
            return {"nodes": []}
            
        # 2. Perform PCA to reduce to 3 dimensions
        X = np.array(vectors)
        # If we have fewer than 3 vectors, we can't do 3D PCA properly, but we'll handle it
        n_components = min(3, len(vectors))
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # 3. Map back to nodes
        result_nodes = []
        for i, node in enumerate(nodes):
            # Normalize coordinates to a reasonable range (e.g., -500 to 500)
            # This helps with visualization scaling
            x = float(X_pca[i, 0]) * 1000
            y = float(X_pca[i, 1]) * 1000 if n_components > 1 else 0
            z = float(X_pca[i, 2]) * 1000 if n_components > 2 else 0
            
            result_nodes.append({
                **node,
                "fx": x, # Fixed positions for force graph
                "fy": y,
                "fz": z,
                "val": 1
            })
            
        return {"nodes": result_nodes}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
