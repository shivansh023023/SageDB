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
page = st.sidebar.selectbox("Navigation", ["Add Data", "Search", "Graph View", "System Health"])

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
            weight = st.slider("Weight", 0.0, 1.0, 0.5)
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
                        st.success("Edge Created Successfully")
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

# --- Page: Search ---
elif page == "Search":
    st.header("Hybrid Search")
    
    with st.form("search_form"):
        query = st.text_input("Search Query")
        col1, col2, col3 = st.columns(3)
        with col1:
            alpha = st.slider("Vector Weight (Alpha)", 0.0, 1.0, 0.7)
        with col2:
            beta = st.slider("Graph Weight (Beta)", 0.0, 1.0, 0.3)
        with col3:
            top_k = st.number_input("Top K", min_value=1, max_value=50, value=5)
            
        search_btn = st.form_submit_button("Search")
        
    if search_btn and query:
        try:
            payload = {
                "text": query,
                "alpha": alpha,
                "beta": beta,
                "top_k": top_k
            }
            res = requests.post(f"{API_URL}/v1/search/hybrid", json=payload)
            
            if res.status_code == 200:
                results = res.json()['results']
                st.write(f"Found {len(results)} results:")
                
                for item in results:
                    with st.expander(f"{item['score']:.4f} | {item['text'][:50]}..."):
                        st.write(f"**UUID:** `{item['uuid']}`")
                        st.write(f"**Text:** {item['text']}")
                        st.write(f"**Scores:** Vector={item['vector_score']:.4f}, Graph={item['graph_score']:.4f}")
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
            # If no start node, try to find one via search or just pick one (not implemented in API yet, so user must provide or we search)
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
                        # Simple NetworkX Plot
                        G = nx.DiGraph()
                        for n in nodes:
                            G.add_node(n)
                        for e in edges:
                            G.add_edge(e['source'], e['target'], label=e['relation'])
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        pos = nx.spring_layout(G)
                        nx.draw(G, pos, with_labels=False, node_color='skyblue', node_size=500, ax=ax)
                        
                        # Draw labels (UUIDs are long, maybe just first few chars)
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
