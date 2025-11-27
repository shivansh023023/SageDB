import streamlit as st
import requests
import json
import networkx as nx
import matplotlib.pyplot as plt

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="SageDB UI", layout="wide")
st.title("SageDB: Vector + Graph Native Database")

# Sidebar Navigation
page = st.sidebar.selectbox("Navigation", ["Search", "Add Data", "Manage Data", "Graph View", "Data Explorer", "System Health"])

def check_health():
    try:
        res = requests.get(f"{API_URL}/health")
        return res.json() if res.status_code == 200 else None
    except:
        return None

# --- Page: System Health ---
if page == "System Health":
    st.header("System Status")
    health = check_health()
    if health:
        st.success("Backend is Online")
        st.json(health)
    else:
        st.error("Backend is Offline. Please start the server.")

# --- Page: Add Data ---
elif page == "Add Data":
    st.header("Add Data")
    
    tab1, tab2 = st.tabs(["Create Node", "Create Edge"])
    
    with tab1:
        st.subheader("Create New Node")
        with st.form("node_form"):
            text = st.text_area("Text Content")
            node_type = st.selectbox("Type", ["document", "entity", "concept"])
            metadata_str = st.text_area("Metadata (JSON)", value='{"category": "general"}')
            submitted = st.form_submit_button("Create Node")
            
            if submitted:
                try:
                    metadata = json.loads(metadata_str)
                    payload = {"text": text, "type": node_type, "metadata": metadata}
                    res = requests.post(f"{API_URL}/v1/nodes", json=payload)
                    if res.status_code == 200:
                        st.success(f"Node Created: {res.json()['uuid']}")
                    else:
                        st.error(f"Error: {res.text}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON for metadata")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    with tab2:
        st.subheader("Create Relationship (Edge)")
        with st.form("edge_form"):
            source_id = st.text_input("Source Node UUID")
            target_id = st.text_input("Target Node UUID")
            relation = st.text_input("Relation Type", value="related_to")
            weight = st.slider("Weight", 0.1, 1.0, 0.5)
            submitted_edge = st.form_submit_button("Create Edge")
            
            if submitted_edge:
                try:
                    payload = {
                        "source_id": source_id, 
                        "target_id": target_id, 
                        "relation": relation, 
                        "weight": weight
                    }
                    res = requests.post(f"{API_URL}/v1/edges", json=payload)
                    if res.status_code == 200:
                        edge_data = res.json()
                        st.success(f"Edge Created! ID: {edge_data['id']}")
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

# --- Page: Manage Data ---
elif page == "Manage Data":
    st.header("Manage Data")
    
    tab1, tab2 = st.tabs(["Update/Delete Node", "Delete Edge"])
    
    with tab1:
        st.subheader("Update or Delete Node")
        node_uuid = st.text_input("Node UUID", key="manage_node_uuid")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Update Node**")
            new_text = st.text_area("New Text (leave empty to keep current)", key="update_text")
            new_metadata_str = st.text_area("New Metadata JSON (leave empty to keep current)", key="update_meta")
            
            if st.button("Update Node"):
                if not node_uuid:
                    st.error("Please enter a Node UUID")
                else:
                    try:
                        payload = {}
                        if new_text.strip():
                            payload["text"] = new_text
                        if new_metadata_str.strip():
                            payload["metadata"] = json.loads(new_metadata_str)
                        
                        if not payload:
                            st.warning("Nothing to update")
                        else:
                            res = requests.put(f"{API_URL}/v1/nodes/{node_uuid}", json=payload)
                            if res.status_code == 200:
                                st.success("Node updated successfully!")
                                st.json(res.json())
                            else:
                                st.error(f"Error: {res.text}")
                    except json.JSONDecodeError:
                        st.error("Invalid JSON for metadata")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")
        
        with col2:
            st.write("**Delete Node**")
            st.warning("⚠️ This will also delete all connected edges!")
            if st.button("Delete Node", type="primary"):
                if not node_uuid:
                    st.error("Please enter a Node UUID")
                else:
                    try:
                        res = requests.delete(f"{API_URL}/v1/nodes/{node_uuid}")
                        if res.status_code == 200:
                            st.success("Node deleted successfully!")
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")
    
    with tab2:
        st.subheader("Delete Edge")
        edge_id = st.number_input("Edge ID", min_value=1, step=1, key="delete_edge_id")
        
        # Show edge details first
        if st.button("Lookup Edge"):
            try:
                res = requests.get(f"{API_URL}/v1/edges/{int(edge_id)}")
                if res.status_code == 200:
                    st.json(res.json())
                else:
                    st.error(f"Edge not found: {res.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
        
        if st.button("Delete Edge", type="primary"):
            try:
                res = requests.delete(f"{API_URL}/v1/edges/{int(edge_id)}")
                if res.status_code == 200:
                    st.success("Edge deleted successfully!")
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# --- Page: Search ---
elif page == "Search":
    st.header("Search")
    
    search_type = st.radio("Search Type", ["Hybrid", "Vector Only", "Graph Only"], horizontal=True)
    
    with st.form("search_form"):
        query = st.text_input("Search Query")
        
        if search_type == "Hybrid":
            st.info("Hybrid Score = (Alpha × Vector Similarity) + (Beta × Graph Score)")
            col1, col2, col3 = st.columns(3)
            with col1:
                alpha = st.slider("Vector Weight (Alpha)", 0.0, 1.0, 0.7)
            with col2:
                beta = st.slider("Graph Weight (Beta)", 0.0, 1.0, 0.3)
            with col3:
                top_k = st.number_input("Top K", min_value=1, max_value=50, value=5)
        elif search_type == "Vector Only":
            st.info("Pure semantic similarity search using embeddings")
            alpha, beta = 1.0, 0.0
            top_k = st.number_input("Top K", min_value=1, max_value=50, value=5)
        else:  # Graph Only
            st.info("Pure graph-based search using connectivity and centrality")
            alpha, beta = 0.0, 1.0
            top_k = st.number_input("Top K", min_value=1, max_value=50, value=5)
            
        search_btn = st.form_submit_button("Search")
        
    if search_btn and query:
        try:
            if search_type == "Vector Only":
                # Use dedicated vector search endpoint
                payload = {"text": query, "top_k": top_k}
                res = requests.post(f"{API_URL}/v1/search/vector", json=payload)
            else:
                # Use hybrid search endpoint
                payload = {"text": query, "alpha": alpha, "beta": beta, "top_k": top_k}
                res = requests.post(f"{API_URL}/v1/search/hybrid", json=payload)
            
            if res.status_code == 200:
                results = res.json()['results']
                st.write(f"Found {len(results)} results:")
                
                for item in results:
                    with st.expander(f"{item['score']:.4f} | {item['text'][:50]}..."):
                        st.write(f"**UUID:** `{item['uuid']}`")
                        st.write(f"**Text:** {item['text']}")
                        
                        # Visualization of the score components
                        st.write("### Scoring Breakdown")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Vector Score", f"{item['vector_score']:.4f}")
                        with col_b:
                            st.metric("Graph Score", f"{item['graph_score']:.4f}")
                        with col_c:
                            st.metric("Final Score", f"{item['score']:.4f}")
                        
                        if search_type == "Hybrid":
                            # Normalize for display
                            total = alpha + beta
                            norm_alpha = alpha / total if total > 0 else 0.5
                            norm_beta = beta / total if total > 0 else 0.5
                            st.write(f"**Calculation:** `{norm_alpha:.2f} × {item['vector_score']:.4f} + {norm_beta:.2f} × {item['graph_score']:.4f}`")
                        
                        st.json(item['metadata'])
            else:
                st.error(f"Search Failed: {res.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# --- Page: Graph View ---
elif page == "Graph View":
    st.header("Graph Visualization")
    
    start_node = st.text_input("Start Node UUID (Leave empty for random/root)")
    depth = st.slider("Traversal Depth", 1, 3, 2)
    
    if st.button("Visualize"):
        try:
            if not start_node:
                st.warning("Please provide a Start Node UUID. You can find one in the Search tab.")
            else:
                params = {"start_id": start_node, "depth": depth}
                res = requests.get(f"{API_URL}/v1/search/graph", params=params)
                
                if res.status_code == 200:
                    data = res.json()
                    nodes = data.get('nodes', [])
                    edges = data.get('edges', [])
                    
                    st.write(f"Subgraph: {len(nodes)} nodes, {len(edges)} edges")
                    
                    if nodes:
                        G = nx.DiGraph()
                        for n in nodes:
                            G.add_node(n)
                        for e in edges:
                            G.add_edge(e['source'], e['target'], label=e['relation'])
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        pos = nx.spring_layout(G)
                        nx.draw(G, pos, with_labels=False, node_color='skyblue', node_size=500, ax=ax)
                        
                        labels = {n: n[:4] for n in G.nodes()}
                        nx.draw_networkx_labels(G, pos, labels, font_size=8)
                        
                        st.pyplot(fig)
                        
                        st.subheader("Raw Data")
                        st.json(data)
                    else:
                        st.info("No nodes found in this subgraph.")
                else:
                    st.error(f"Error: {res.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# --- Page: Data Explorer ---
elif page == "Data Explorer":
    st.header("Data Explorer")
    
    tab1, tab2 = st.tabs(["Nodes", "Edges"])
    
    with tab1:
        st.subheader("All Nodes")
        if st.button("Refresh Nodes"):
            try:
                res = requests.get(f"{API_URL}/v1/nodes")
                if res.status_code == 200:
                    nodes = res.json()
                    st.write(f"Total Nodes: {len(nodes)}")
                    st.dataframe(nodes)
                else:
                    st.error(f"Error fetching nodes: {res.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
                
    with tab2:
        st.subheader("All Edges")
        if st.button("Refresh Edges"):
            try:
                res = requests.get(f"{API_URL}/v1/edges")
                if res.status_code == 200:
                    edges = res.json()
                    st.write(f"Total Edges: {len(edges)}")
                    st.dataframe(edges)
                else:
                    st.error(f"Error fetching edges: {res.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
