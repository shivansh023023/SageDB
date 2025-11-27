import streamlit as st
import requests
import json
import networkx as nx
import matplotlib.pyplot as plt

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="SageDB UI", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern "New Gen" CSS Styling
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* GLOBAL: Hide ALL Material Symbol text placeholders */
    .material-symbols-rounded,
    span[class*="material-symbols"],
    [data-baseweb="select"] span[class*="material"],
    [data-baseweb="popover"] span[class*="material"],
    .stExpander span[data-testid="stExpanderToggleIcon"],
    .stExpander [data-testid="stExpanderToggleIcon"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        font-size: 0 !important;
    }
    
    /* Hide any element containing "keyboard_arrow" text */
    [class*="IconContainer"],
    [data-testid*="Icon"],
    summary > div > span:first-child {
        display: none !important;
    }
    
    /* Global font */
    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container - dark theme to match sidebar */
    .stApp {
        background: linear-gradient(135deg, #1a1f2e 0%, #0f172a 50%, #1e293b 100%) !important;
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    
    /* Reduce top padding/margin */
    .stApp > header {
        background: transparent !important;
    }
    .stMainBlockContainer {
        padding-top: 1rem !important;
    }
    
    /* Tighter spacing for elements */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    .stMarkdown {
        margin-bottom: 0.25rem !important;
    }
    
    /* Gradient background for sidebar - seamless transition */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 50%, #252d3d 100%) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.2) !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.3) !important;
        min-width: 220px !important;
        padding-top: 0 !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0.5rem !important;
    }
    section[data-testid="stSidebar"] > div > div {
        padding-top: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0.5rem !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
        transition: all 0.3s ease;
        padding: 6px 10px !important;
        border-radius: 8px !important;
        margin-bottom: 2px !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #ffffff !important;
        background: rgba(99, 102, 241, 0.15) !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Main content area text colors for dark theme */
    h1 {
        color: #f1f5f9 !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    h2, h3, h4 {
        color: #e2e8f0 !important;
        font-weight: 600;
    }
    p, span, label, .stMarkdown {
        color: #cbd5e1 !important;
    }
    
    /* Glassmorphism cards - darker theme */
    .stExpander {
        background: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .stExpander summary p {
        color: #e2e8f0 !important;
    }
    
    /* AGGRESSIVELY hide expander arrow icons and text */
    .stExpander [data-testid="stExpanderToggleIcon"],
    .stExpander summary svg,
    .stExpander summary > div > span:first-child,
    .stExpander details summary span[class*="material"],
    details summary::marker,
    details summary::-webkit-details-marker,
    [data-testid="stExpander"] summary > div:first-child > span:first-child {
        display: none !important;
        visibility: hidden !important;
        font-size: 0 !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }
    
    /* Style expander header - add a simple indicator */
    .stExpander summary {
        padding-left: 0.5rem !important;
    }
    .stExpander summary > div > p {
        margin-left: 0 !important;
    }
    /* Add a simple +/- indicator via CSS */
    .stExpander summary::before {
        content: "▸ " !important;
        font-size: 12px !important;
        color: #6366f1 !important;
    }
    .stExpander[open] summary::before {
        content: "▾ " !important;
    }
    
    /* Modern buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
        border: none;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    }
    
    /* Metrics - dark theme */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f1f5f9 !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: #94a3b8 !important;
    }
    
    /* Modern alerts - dark theme */
    .stAlert {
        border-radius: 12px;
        border: none;
        background: rgba(30, 41, 59, 0.8) !important;
    }
    .stSuccess {
        background: rgba(16, 185, 129, 0.15) !important;
        border-left: 4px solid #10b981 !important;
    }
    .stError {
        background: rgba(239, 68, 68, 0.15) !important;
        border-left: 4px solid #ef4444 !important;
    }
    .stWarning {
        background: rgba(245, 158, 11, 0.15) !important;
        border-left: 4px solid #f59e0b !important;
    }
    .stInfo {
        background: rgba(99, 102, 241, 0.15) !important;
        border-left: 4px solid #6366f1 !important;
    }
    
    /* Tabs styling - dark theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.5) !important;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        color: #94a3b8 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2) !important;
        color: #f1f5f9 !important;
    }
    
    /* Input fields - dark theme */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #334155 !important;
        background: rgba(30, 41, 59, 0.8) !important;
        color: #f1f5f9 !important;
        transition: border-color 0.2s ease;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #64748b !important;
    }
    
    /* Dividers - dark theme */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent) !important;
        margin: 1.5rem 0;
    }
    .stDivider {
        background: linear-gradient(90deg, transparent, #334155, transparent) !important;
    }
    
    /* Dataframe styling - dark theme */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: rgba(30, 41, 59, 0.8) !important;
    }
    
    /* Select boxes - dark theme */
    [data-baseweb="select"] {
        background: rgba(30, 41, 59, 0.8) !important;
    }
    [data-baseweb="select"] > div {
        background: rgba(30, 41, 59, 0.8) !important;
        border-color: #334155 !important;
    }
    
    /* Number inputs - dark theme */
    .stNumberInput > div > div > input {
        background: rgba(30, 41, 59, 0.8) !important;
        border-color: #334155 !important;
        color: #f1f5f9 !important;
    }
    
    /* Sliders - purple accent */
    .stSlider [data-baseweb="slider"] {
        background: #334155 !important;
    }
    .stSlider [role="slider"] {
        background: #6366f1 !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: transparent !important;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #cbd5e1 !important;
    }
    
    /* File uploader - dark theme */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 2px dashed #334155 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #6366f1 !important;
    }
    
    /* Forms - dark theme, compact */
    [data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    
    /* Compact dividers */
    hr, [data-testid="stDivider"] {
        margin: 0.75rem 0 !important;
    }
    
    /* Compact headers */
    h1 {
        margin-bottom: 0.25rem !important;
        font-size: 1.75rem !important;
    }
    h2, h3 {
        margin-bottom: 0.25rem !important;
        margin-top: 0.5rem !important;
    }
    h4 {
        margin-bottom: 0.25rem !important;
        margin-top: 0.25rem !important;
        font-size: 1rem !important;
    }
    
    /* Compact caption */
    .stCaption, [data-testid="stCaptionContainer"] {
        margin-bottom: 0.5rem !important;
    }
    
    /* Compact expanders */
    .stExpander {
        margin-bottom: 0.5rem !important;
    }
    .stExpander details {
        padding: 0.5rem !important;
    }
    
    /* Compact metrics */
    [data-testid="stMetric"] {
        padding: 0.5rem !important;
    }
    
    /* Compact tabs */
    .stTabs {
        margin-bottom: 0.5rem !important;
    }
    
    /* === HIDE SIDEBAR COLLAPSE BUTTONS COMPLETELY === */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="baseButton-header"],
    button[kind="header"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header[data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 SageDB")
st.caption("Vector + Graph Native Database")

# Sidebar Navigation
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🔍 Search",
        "📁 Ingest Files", 
        "➕ Add Data",
        "⚙️ Manage Data",
        "🕸️ Graph View",
        "📊 Data Explorer",
        "💚 System Health"
    ],
    label_visibility="collapsed"
)

def check_health():
    try:
        res = requests.get(f"{API_URL}/health")
        return res.json() if res.status_code == 200 else None
    except:
        return None

# --- Page: System Health ---
if page == "💚 System Health":
    st.header("💚 System Health")
    st.caption("Monitor your database status and system metrics")
    
    health = check_health()
    
    if health:
        # Status indicator
        st.success("🟢 All systems operational")
        
        st.divider()
        
        # Metrics cards
        st.markdown("#### 📊 Database Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📦 Total Nodes",
                value=health.get('nodes', 0),
                delta=None
            )
        with col2:
            st.metric(
                label="🔗 Total Edges",
                value=health.get('edges', 0),
                delta=None
            )
        with col3:
            st.metric(
                label="📐 Vector Dimensions",
                value=health.get('vector_dim', 384),
                delta=None
            )
        with col4:
            st.metric(
                label="⚡ Status",
                value="Online",
                delta=None
            )
        
        st.divider()
        
        # Server info
        col_info, col_raw = st.columns([1, 1])
        
        with col_info:
            st.markdown("#### 🖥️ Server Information")
            st.markdown(f"""
            | Property | Value |
            |----------|-------|
            | API URL | `{API_URL}` |
            | Status | ✅ Connected |
            | Version | 1.0.0 |
            """)
            
        with col_raw:
            st.markdown("#### 📄 Raw Response")
            with st.expander("View JSON", expanded=False):
                st.json(health)
        
        st.divider()
        
        # Quick actions
        st.markdown("#### ⚡ Quick Actions")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.rerun()
        with col_b:
            if st.button("📊 View Data Explorer", use_container_width=True):
                st.info("Navigate using the sidebar →")
        with col_c:
            if st.button("🔍 Go to Search", use_container_width=True):
                st.info("Navigate using the sidebar →")
    else:
        st.error("🔴 Backend is Offline")
        st.warning("""
        **Unable to connect to the SageDB server.**
        
        Please ensure the backend is running:
        ```bash
        python main.py
        ```
        
        The server should be running at `http://localhost:8000`
        """)
        
        if st.button("🔄 Retry Connection", type="primary"):
            st.rerun()

# --- Page: Ingest Files ---
elif page == "📁 Ingest Files":
    st.header("📁 Ingest Documents")
    st.caption("Upload and process text-based files into the knowledge graph")
    
    # Status bar
    col_status, col_formats = st.columns([1, 2])
    with col_status:
        health = check_health()
        if health:
            st.success(f"✅ Backend Online • {health.get('nodes', 0)} nodes")
        else:
            st.error("❌ Backend Offline")
    
    with col_formats:
        st.info("📄 Supported: `.md` `.txt` `.html` `.json` `.xml`")
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📄 Single File", "📚 Batch Upload", "📝 Raw Text"])
    
    with tab1:
        st.markdown("#### Upload a Single File")
        
        uploaded_file = st.file_uploader(
            "Drag and drop or click to browse", 
            type=['md', 'txt', 'html', 'htm', 'json', 'xml'],
            key="single_upload",
            help="Select a text-based file to ingest"
        )
        
        if uploaded_file:
            st.caption(f"📎 Selected: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
        
        with st.expander("⚙️ Options"):
            create_seq_edges = st.checkbox(
                "Create sequential edges between chunks", 
                value=True, 
                key="single_seq",
                help="Links consecutive chunks with 'next_chunk' relationships"
            )
        
        if st.button("🚀 Ingest File", type="primary", use_container_width=True, disabled=not uploaded_file):
            if uploaded_file is not None:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                        data = {"create_sequential_edges": str(create_seq_edges).lower()}
                        res = requests.post(f"{API_URL}/v1/ingest/file", files=files, data=data)
                        
                        if res.status_code == 200:
                            result = res.json()
                            if result['success']:
                                st.success(f"✅ {result['message']}")
                                
                                # Results in a nice grid
                                cols = st.columns(4)
                                cols[0].metric("📄 Doc ID", result.get('document_id', 'N/A')[:8] + "...")
                                cols[1].metric("🧩 Chunks", result.get('chunks_created', 0))
                                cols[2].metric("🔵 Nodes", result.get('nodes_created', 0))
                                cols[3].metric("🔗 Edges", result.get('edges_created', 0))
                                
                                if result.get('processing_time_ms'):
                                    st.caption(f"⏱️ Processed in {result['processing_time_ms']:.0f}ms")
                            else:
                                st.error(f"❌ {result['message']}")
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")
    
    with tab2:
        st.markdown("#### Upload Multiple Files")
        
        uploaded_files = st.file_uploader(
            "Drag and drop multiple files",
            type=['md', 'txt', 'html', 'htm', 'json', 'xml'],
            accept_multiple_files=True,
            key="batch_upload",
            help="Select multiple files for batch processing"
        )
        
        if uploaded_files:
            st.caption(f"📎 Selected: **{len(uploaded_files)} files**")
        
        with st.expander("⚙️ Options"):
            create_seq_edges_batch = st.checkbox(
                "Create sequential edges between chunks", 
                value=True, 
                key="batch_seq"
            )
        
        if st.button("🚀 Ingest All Files", type="primary", use_container_width=True, disabled=not uploaded_files):
            progress = st.progress(0, text="Starting batch ingestion...")
            
            with st.spinner(f"Processing {len(uploaded_files)} files..."):
                try:
                    files = [("files", (f.name, f.getvalue())) for f in uploaded_files]
                    data = {"create_sequential_edges": str(create_seq_edges_batch).lower()}
                    res = requests.post(f"{API_URL}/v1/ingest/batch", files=files, data=data)
                    
                    progress.progress(100, text="Complete!")
                    
                    if res.status_code == 200:
                        batch_result = res.json()
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("✅ Succeeded", batch_result['successful'])
                        col2.metric("❌ Failed", batch_result['failed'])
                        col3.metric("📊 Total", batch_result['total_files'])
                        
                        # Individual results
                        st.divider()
                        for result in batch_result['results']:
                            icon = "✅" if result['success'] else "❌"
                            with st.expander(f"{icon} {result['filename']}"):
                                if result['success']:
                                    st.markdown(f"**Chunks:** {result['chunks_created']} • **Nodes:** {result['nodes_created']} • **Edges:** {result['edges_created']}")
                                else:
                                    st.error(result['message'])
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
    
    with tab3:
        st.markdown("#### Paste Raw Text")
        
        with st.form("text_ingest_form"):
            text_content = st.text_area(
                "Text Content", 
                height=200,
                placeholder="Paste your text here..."
            )
            
            col1, col2 = st.columns(2)
            with col1:
                filename = st.text_input("Document Name", value="input.txt", help="Name for this document")
            with col2:
                file_type = st.selectbox("Format", ["txt", "md", "html", "json"])
            
            create_seq_edges_text = st.checkbox("Create sequential edges", value=True, key="text_seq")
            
            submitted = st.form_submit_button("🚀 Ingest Text", type="primary", use_container_width=True)
            
            if submitted:
                if not text_content.strip():
                    st.warning("Please enter some text to ingest")
                else:
                    try:
                        data = {
                            "text": text_content,
                            "filename": filename,
                            "file_type": file_type,
                            "create_sequential_edges": str(create_seq_edges_text).lower()
                        }
                        res = requests.post(f"{API_URL}/v1/ingest/text", data=data)
                        
                        if res.status_code == 200:
                            result = res.json()
                            if result['success']:
                                st.success(f"✅ {result['message']}")
                                st.markdown(f"**Created:** {result['chunks_created']} chunks • {result['nodes_created']} nodes • {result['edges_created']} edges")
                            else:
                                st.error(f"❌ {result['message']}")
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

# --- Page: Add Data ---
elif page == "➕ Add Data":
    st.header("➕ Add Data")
    st.caption("Create new nodes and relationships in your knowledge graph")
    
    tab1, tab2 = st.tabs(["🔵 Create Node", "🔗 Create Edge"])
    
    with tab1:
        st.markdown("#### Create a New Node")
        st.markdown("Add a piece of knowledge to the database. It will be automatically embedded for semantic search.")
        
        with st.form("node_form"):
            text = st.text_area(
                "📝 Text Content", 
                height=150,
                placeholder="Enter the text content for this node..."
            )
            
            col1, col2 = st.columns(2)
            with col1:
                node_type = st.selectbox(
                    "🏷️ Node Type", 
                    ["document", "entity", "concept"],
                    help="Categorize this node"
                )
            with col2:
                category = st.text_input("📂 Category", value="general")
            
            with st.expander("🔧 Advanced: Custom Metadata (JSON)"):
                metadata_str = st.text_area(
                    "Metadata", 
                    value='{"category": "general"}',
                    help="Add custom key-value pairs"
                )
            
            submitted = st.form_submit_button("✨ Create Node", type="primary", use_container_width=True)
            
            if submitted:
                if not text.strip():
                    st.warning("Please enter some text content")
                else:
                    try:
                        metadata = json.loads(metadata_str)
                        metadata["category"] = category
                        payload = {"text": text, "type": node_type, "metadata": metadata}
                        res = requests.post(f"{API_URL}/v1/nodes", json=payload)
                        if res.status_code == 200:
                            result = res.json()
                            st.success("✅ Node created successfully!")
                            st.code(result['uuid'], language=None)
                            st.caption("Copy this UUID to create edges")
                        else:
                            st.error(f"Error: {res.text}")
                    except json.JSONDecodeError:
                        st.error("Invalid JSON in metadata")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

    with tab2:
        st.markdown("#### Create a Relationship")
        st.markdown("Connect two nodes with a weighted edge to build your knowledge graph.")
        
        with st.form("edge_form"):
            col1, col2 = st.columns(2)
            with col1:
                source_id = st.text_input("🔵 Source Node UUID", placeholder="Paste UUID here...")
            with col2:
                target_id = st.text_input("🎯 Target Node UUID", placeholder="Paste UUID here...")
            
            col3, col4 = st.columns(2)
            with col3:
                relation = st.selectbox(
                    "🔗 Relation Type",
                    ["related_to", "part_of", "causes", "depends_on", "similar_to", "opposite_of", "custom"],
                    help="Type of relationship"
                )
                if relation == "custom":
                    relation = st.text_input("Custom relation name")
            with col4:
                weight = st.slider("⚖️ Weight", 0.1, 1.0, 0.5, help="Strength of the relationship")
            
            submitted_edge = st.form_submit_button("🔗 Create Edge", type="primary", use_container_width=True)
            
            if submitted_edge:
                if not source_id or not target_id:
                    st.warning("Please enter both source and target UUIDs")
                else:
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
                            st.success(f"✅ Edge created! ID: {edge_data['id']}")
                            st.markdown(f"**{source_id[:8]}...** → `{relation}` → **{target_id[:8]}...**")
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

# --- Page: Manage Data ---
elif page == "⚙️ Manage Data":
    st.header("⚙️ Manage Data")
    st.caption("Update or delete existing nodes and edges")
    
    tab1, tab2 = st.tabs(["🔵 Manage Nodes", "🔗 Manage Edges"])
    
    with tab1:
        st.markdown("#### Node Operations")
        
        node_uuid = st.text_input(
            "🔍 Enter Node UUID", 
            key="manage_node_uuid",
            placeholder="Paste the node UUID here..."
        )
        
        if node_uuid:
            # Try to fetch current node data
            try:
                res = requests.get(f"{API_URL}/v1/nodes/{node_uuid}")
                if res.status_code == 200:
                    current_node = res.json()
                    st.success("✅ Node found!")
                    with st.expander("📄 Current Node Data", expanded=True):
                        st.json(current_node)
                else:
                    st.warning("Node not found. Check the UUID.")
                    current_node = None
            except:
                current_node = None
                st.info("Enter a UUID and ensure backend is running")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### ✏️ Update Node")
            new_text = st.text_area(
                "New Text", 
                key="update_text",
                placeholder="Leave empty to keep current text",
                height=100
            )
            new_metadata_str = st.text_area(
                "New Metadata (JSON)", 
                key="update_meta",
                placeholder='{"key": "value"}',
                height=80
            )
            
            if st.button("💾 Update Node", use_container_width=True):
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
                                st.success("✅ Node updated!")
                                st.rerun()
                            else:
                                st.error(f"Error: {res.text}")
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")
        
        with col2:
            st.markdown("##### 🗑️ Delete Node")
            st.error("⚠️ **Danger Zone**\n\nThis will permanently delete the node and all connected edges!")
            
            confirm = st.checkbox("I understand this action cannot be undone")
            
            if st.button("🗑️ Delete Node", type="primary", use_container_width=True, disabled=not confirm):
                if not node_uuid:
                    st.error("Please enter a Node UUID")
                else:
                    try:
                        res = requests.delete(f"{API_URL}/v1/nodes/{node_uuid}")
                        if res.status_code == 200:
                            st.success("✅ Node deleted!")
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")
    
    with tab2:
        st.markdown("#### Edge Operations")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            edge_id = st.number_input("🔍 Edge ID", min_value=1, step=1, key="delete_edge_id")
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("🔎 Lookup", use_container_width=True):
                try:
                    res = requests.get(f"{API_URL}/v1/edges/{int(edge_id)}")
                    if res.status_code == 200:
                        st.session_state['found_edge'] = res.json()
                    else:
                        st.error(f"Edge not found")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
        
        if 'found_edge' in st.session_state:
            edge = st.session_state['found_edge']
            with st.expander("📄 Edge Details", expanded=True):
                st.json(edge)
        
        st.divider()
        
        confirm_edge = st.checkbox("I want to delete this edge", key="confirm_edge_delete")
        if st.button("🗑️ Delete Edge", type="primary", disabled=not confirm_edge):
            try:
                res = requests.delete(f"{API_URL}/v1/edges/{int(edge_id)}")
                if res.status_code == 200:
                    st.success("✅ Edge deleted!")
                    if 'found_edge' in st.session_state:
                        del st.session_state['found_edge']
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# --- Page: Search ---
elif page == "🔍 Search":
    st.header("🔍 Search")
    st.caption("Find nodes using semantic similarity and graph relationships")
    
    # Search type selection with better styling
    col_type, col_space = st.columns([2, 3])
    with col_type:
        search_type = st.radio("Search Type", ["Hybrid", "Vector Only", "Graph Only"], horizontal=True)
    
    st.divider()
    
    with st.form("search_form"):
        query = st.text_input("🔎 Enter your search query", placeholder="Type something to search...")
        
        # Settings in expandable section
        with st.expander("⚙️ Search Settings", expanded=(search_type == "Hybrid")):
            if search_type == "Hybrid":
                st.caption("Hybrid Score = (Alpha × Vector) + (Beta × Graph)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    alpha = st.slider("Vector Weight (α)", 0.0, 1.0, 0.7)
                with col2:
                    beta = st.slider("Graph Weight (β)", 0.0, 1.0, 0.3)
                with col3:
                    top_k = st.number_input("Results", min_value=1, max_value=50, value=5)
            elif search_type == "Vector Only":
                st.caption("Pure semantic similarity using embeddings")
                alpha, beta = 1.0, 0.0
                top_k = st.number_input("Results", min_value=1, max_value=50, value=5)
            else:  # Graph Only
                st.caption("Graph-based search using connectivity")
                alpha, beta = 0.0, 1.0
                top_k = st.number_input("Results", min_value=1, max_value=50, value=5)
            
        search_btn = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)
        
    if search_btn and query:
        try:
            with st.spinner("Searching..."):
                if search_type == "Vector Only":
                    payload = {"text": query, "top_k": top_k}
                    res = requests.post(f"{API_URL}/v1/search/vector", json=payload)
                else:
                    payload = {"text": query, "alpha": alpha, "beta": beta, "top_k": top_k}
                    res = requests.post(f"{API_URL}/v1/search/hybrid", json=payload)
            
            if res.status_code == 200:
                results = res.json()['results']
                
                if results:
                    st.success(f"Found {len(results)} results")
                    st.divider()
                    
                    for i, item in enumerate(results, 1):
                        with st.container():
                            # Result header with rank and score
                            col_rank, col_content = st.columns([1, 10])
                            with col_rank:
                                st.markdown(f"### #{i}")
                            with col_content:
                                st.markdown(f"**Score:** `{item['score']:.4f}`")
                                st.markdown(f"**Text:** {item['text'][:200]}{'...' if len(item['text']) > 200 else ''}")
                            
                            # Expandable details
                            with st.expander("View Details"):
                                st.code(item['uuid'], language=None)
                                
                                # Score breakdown
                                st.markdown("**Score Breakdown**")
                                score_cols = st.columns(3)
                                with score_cols[0]:
                                    st.metric("Vector", f"{item['vector_score']:.4f}")
                                with score_cols[1]:
                                    st.metric("Graph", f"{item['graph_score']:.4f}")
                                with score_cols[2]:
                                    st.metric("Final", f"{item['score']:.4f}")
                                
                                if item.get('metadata'):
                                    st.markdown("**Metadata**")
                                    st.json(item['metadata'])
                            
                            st.divider()
                else:
                    st.info("No results found. Try a different query.")
            else:
                st.error(f"Search Failed: {res.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# --- Page: Graph View ---
elif page == "🕸️ Graph View":
    st.header("🕸️ Graph Visualization")
    st.caption("Explore relationships between nodes in your knowledge graph")
    
    st.divider()
    
    # Configuration section
    st.markdown("#### 🎯 Visualization Settings")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        start_node = st.text_input(
            "🔑 Start Node UUID",
            placeholder="Enter a node UUID from Search results...",
            help="The UUID of the node to start graph traversal from"
        )
    
    with col2:
        depth = st.selectbox(
            "📏 Depth",
            options=[1, 2, 3],
            index=1,
            help="How many levels of connections to show"
        )
    
    # Visualization options
    with st.expander("⚙️ Display Options", expanded=False):
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            node_color = st.color_picker("Node Color", "#667eea")
            node_size = st.slider("Node Size", 200, 1000, 500)
        with col_opt2:
            show_labels = st.checkbox("Show Labels", value=True)
            fig_width = st.slider("Figure Width", 8, 16, 12)
    
    st.divider()
    
    if st.button("🚀 Visualize Graph", type="primary", use_container_width=True):
        if not start_node:
            st.warning("⚠️ Please provide a Start Node UUID. You can find one in the **🔍 Search** tab.")
        else:
            try:
                with st.spinner("Loading graph data..."):
                    params = {"start_id": start_node, "depth": depth}
                    res = requests.get(f"{API_URL}/v1/search/graph", params=params)
                
                if res.status_code == 200:
                    data = res.json()
                    nodes = data.get('nodes', [])
                    edges = data.get('edges', [])
                    
                    # Stats bar
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.metric("📦 Nodes", len(nodes))
                    with col_s2:
                        st.metric("🔗 Edges", len(edges))
                    with col_s3:
                        st.metric("📏 Depth", depth)
                    
                    if nodes:
                        st.markdown("#### 📈 Graph Visualization")
                        
                        G = nx.DiGraph()
                        for n in nodes:
                            G.add_node(n)
                        for e in edges:
                            G.add_edge(e['source'], e['target'], label=e['relation'])
                        
                        fig, ax = plt.subplots(figsize=(fig_width, 8))
                        fig.patch.set_facecolor('#0e1117')
                        ax.set_facecolor('#0e1117')
                        
                        pos = nx.spring_layout(G, k=2, iterations=50)
                        nx.draw_networkx(
                            G, pos, 
                            with_labels=False, 
                            node_color=node_color, 
                            node_size=node_size, 
                            ax=ax,
                            edge_color='#4a5568',
                            width=2,
                            arrows=True,
                            arrowsize=15
                        )
                        
                        if show_labels:
                            labels = {n: n[:6] + "..." for n in G.nodes()}
                            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='white')
                        
                        # Draw edge labels
                        edge_labels = nx.get_edge_attributes(G, 'label')
                        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, font_color='#a0aec0')
                        
                        st.pyplot(fig)
                        
                        st.divider()
                        
                        # Raw data section
                        with st.expander("📄 Raw Graph Data", expanded=False):
                            tab_n, tab_e = st.tabs(["Nodes", "Edges"])
                            with tab_n:
                                st.json(nodes)
                            with tab_e:
                                st.json(edges)
                    else:
                        st.info("📭 No nodes found in this subgraph. Try a different UUID or increase depth.")
                else:
                    st.error(f"❌ Error: {res.text}")
            except Exception as e:
                st.error(f"❌ Connection Error: {e}")

# --- Page: Data Explorer ---
elif page == "📊 Data Explorer":
    st.header("📊 Data Explorer")
    st.caption("Browse and explore all data stored in your database")
    
    # Quick stats
    try:
        health = check_health()
        if health:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📦 Total Nodes", health.get('nodes', 0))
            with col2:
                st.metric("🔗 Total Edges", health.get('edges', 0))
            with col3:
                st.metric("⚡ Status", "Online")
    except:
        pass
    
    st.divider()
    
    tab1, tab2 = st.tabs(["📦 Nodes", "🔗 Edges"])
    
    with tab1:
        st.markdown("#### 📦 Node Browser")
        
        col_btn, col_search = st.columns([1, 3])
        with col_btn:
            load_nodes = st.button("🔄 Load Nodes", type="primary", use_container_width=True)
        with col_search:
            node_filter = st.text_input("🔍 Filter by UUID", placeholder="Type to filter...")
        
        if load_nodes or 'cached_nodes' in st.session_state:
            if load_nodes:
                try:
                    with st.spinner("Loading nodes..."):
                        res = requests.get(f"{API_URL}/v1/nodes")
                    if res.status_code == 200:
                        st.session_state['cached_nodes'] = res.json()
                    else:
                        st.error(f"❌ Error fetching nodes: {res.text}")
                except Exception as e:
                    st.error(f"❌ Connection Error: {e}")
            
            if 'cached_nodes' in st.session_state:
                nodes = st.session_state['cached_nodes']
                
                # Apply filter
                if node_filter:
                    nodes = [n for n in nodes if node_filter.lower() in str(n).lower()]
                
                st.success(f"✅ Showing {len(nodes)} nodes")
                
                if nodes:
                    # Convert to dataframe for better display
                    import pandas as pd
                    df = pd.DataFrame(nodes)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Export option
                    with st.expander("📥 Export Options"):
                        st.download_button(
                            label="📥 Download as CSV",
                            data=df.to_csv(index=False),
                            file_name="sagedb_nodes.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("📭 No nodes match your filter.")
                
    with tab2:
        st.markdown("#### 🔗 Edge Browser")
        
        col_btn2, col_search2 = st.columns([1, 3])
        with col_btn2:
            load_edges = st.button("🔄 Load Edges", type="primary", use_container_width=True)
        with col_search2:
            edge_filter = st.text_input("🔍 Filter by relation", placeholder="Type to filter...")
        
        if load_edges or 'cached_edges' in st.session_state:
            if load_edges:
                try:
                    with st.spinner("Loading edges..."):
                        res = requests.get(f"{API_URL}/v1/edges")
                    if res.status_code == 200:
                        st.session_state['cached_edges'] = res.json()
                    else:
                        st.error(f"❌ Error fetching edges: {res.text}")
                except Exception as e:
                    st.error(f"❌ Connection Error: {e}")
            
            if 'cached_edges' in st.session_state:
                edges = st.session_state['cached_edges']
                
                # Apply filter
                if edge_filter:
                    edges = [e for e in edges if edge_filter.lower() in str(e).lower()]
                
                st.success(f"✅ Showing {len(edges)} edges")
                
                if edges:
                    # Convert to dataframe for better display
                    import pandas as pd
                    df = pd.DataFrame(edges)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Export option
                    with st.expander("📥 Export Options"):
                        st.download_button(
                            label="📥 Download as CSV",
                            data=df.to_csv(index=False),
                            file_name="sagedb_edges.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("📭 No edges match your filter.")
