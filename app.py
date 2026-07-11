import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="RAGBench: Evaluation-Driven RAG QA System",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Custom CSS for premium design
st.markdown("""
<style>
    .reportview-container {
        background: #0F172A;
    }
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    h1, h2, h3 {
        color: #F1F5F9 !important;
    }
    .stButton>button {
        background-color: #0F766E;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0D9488;
        border: none;
        box-shadow: 0px 4px 10px rgba(13, 148, 136, 0.4);
    }
    .chunk-card {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0D9488;
        margin-bottom: 12px;
        color: #E2E8F0;
    }
    .chunk-meta {
        font-size: 0.85em;
        color: #94A3B8;
        margin-bottom: 8px;
        font-weight: bold;
    }
    .answer-card {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 20px;
        line-height: 1.6;
        color: #F8FAFC;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-val {
        font-size: 2.2em;
        font-weight: bold;
        color: #38BDF8;
        margin-top: 5px;
    }
    .metric-lbl {
        font-size: 0.85em;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_exists=True, unsafe_allow_html=True)

# Helper function to check if API is running
def check_api():
    try:
        response = requests.get(f"{API_URL}/")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

# Sidebar layout
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=70)
    st.title("RAGBench Settings")
    st.write("An Evaluation-Driven RAG QA System with Automated Ragas Benchmarking.")
    st.markdown("---")
    
    # API status
    api_online = check_api()
    if api_online:
        st.success("API Status: Connected")
    else:
        st.error("API Status: Disconnected")
        st.warning("Please start the FastAPI backend:\n`uvicorn api:app --reload`")
        
    st.markdown("### Document Ingestion")
    uploaded_file = st.file_uploader("Upload Policy PDF", type=["pdf"])
    
    chunk_size = st.slider("Chunk Size (chars)", min_value=200, max_value=2000, value=500, step=50)
    chunk_overlap = st.slider("Chunk Overlap (chars)", min_value=0, max_value=500, value=50, step=10)
    
    if st.button("Upload & Ingest", disabled=not api_online or uploaded_file is None):
        with st.spinner("Uploading and indexing PDF..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                params = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
                response = requests.post(f"{API_URL}/upload", files=files, params=params)
                if response.status_code == 200:
                    st.success(response.json()["message"])
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
                
    st.markdown("---")
    st.markdown("### Retrieval Settings")
    alpha = st.slider("Retrieval Alpha (Vector vs Keyword)", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                      help="0.0: Pure Keyword (BM25)\n1.0: Pure Semantic (ChromaDB)\n0.5: Hybrid (RRF)")
    top_n = st.slider("Retrieve Top K Chunks", min_value=1, max_value=10, value=5, step=1)
    
    # API key warning
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not gemini_key and not openai_key:
        st.warning("⚠️ No API Key found. Add GEMINI_API_KEY or OPENAI_API_KEY in your .env file.")

# Main app tabs
tabs = st.tabs(["🔍 RAG QA Portal", "📊 Ragas Evaluation Dashboard"])

# TAB 1: RAG QA Portal
with tabs[0]:
    st.markdown("# 🔍 RAGBench QA Portal")
    st.markdown("Ask natural language questions about your uploaded employee handbook, leave policy, or code of conduct.")
    
    query = st.text_input("Ask a question about HR policies:", placeholder="e.g., How many vacation days can I carry over?", key="query_input")
    
    if query:
        if not api_online:
            st.error("Cannot query. FastAPI backend is not running.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    payload = {"query": query, "top_n": top_n, "alpha": alpha}
                    response = requests.post(f"{API_URL}/query", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            st.markdown("### 🤖 Generated Answer")
                            st.markdown(f'<div class="answer-card">{data["answer"]}</div>', unsafe_allow_html=True)
                            
                            # Metrics block
                            latency_ms = data.get("retrieve_latency_ms", 0)
                            st.markdown(f"**Retrieval Latency:** `{latency_ms} ms` | **Context Source Chunks:** `{len(data.get('sources', []))}`")
                        
                        with col2:
                            st.markdown("### 📄 Retrieved Context Chunks")
                            sources = data.get("sources", [])
                            if not sources:
                                st.info("No sources retrieved.")
                            else:
                                for s in sources:
                                    st.markdown(f"""
                                    <div class="chunk-card">
                                        <div class="chunk-meta">Rank {s['rank']} | Source: {s['source_name']} | Page: {s['page']}</div>
                                        <div>{s['content']}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.error(f"Error querying backend: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to query: {e}")

# TAB 2: Ragas Evaluation Dashboard
with tabs[1]:
    st.markdown("# 📊 Ragas Automated Evaluation Dashboard")
    st.markdown("This dashboard tracks retrieval quality, model faithfulness, and correctness scores against a golden evaluation dataset of 20 ground-truth questions.")
    
    # Load metrics
    metrics_data = None
    if api_online:
        try:
            resp = requests.get(f"{API_URL}/metrics")
            if resp.status_code == 200 and resp.json().get("status") == "success":
                metrics_data = resp.json()["metrics"]
        except Exception as e:
            pass
            
    # Evaluation triggers
    col_trigger_1, col_trigger_2 = st.columns([2, 3])
    with col_trigger_1:
        if st.button("🚀 Run Ragas Evaluation Dataset", disabled=not api_online):
            with st.spinner("Running evaluation over 20 golden QA pairs. This evaluates Faithfulness, Relevancy, Recall, and Precision using Ragas. Please wait (takes ~1-2 mins)..."):
                try:
                    resp = requests.post(f"{API_URL}/evaluate", params={"top_n": top_n, "alpha": alpha})
                    if resp.status_code == 200:
                        st.success("Evaluation complete! Reloading metrics...")
                        # Force refresh
                        resp_metrics = requests.get(f"{API_URL}/metrics")
                        if resp_metrics.status_code == 200 and resp_metrics.json().get("status") == "success":
                            metrics_data = resp_metrics.json()["metrics"]
                    else:
                        st.error(f"Evaluation failed: {resp.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to trigger evaluation: {e}")
                    
    with col_trigger_2:
        st.info("💡 Running evaluation queries the system with 20 pre-defined employee policy questions, compiles the retrieved contexts and generated answers, and computes four Ragas validation scores.")
        
    if metrics_data:
        st.markdown("### Average Ragas Evaluation Scores")
        
        # Grid of metric cards
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Faithfulness</div>
                <div class="metric-val">{metrics_data['faithfulness']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Answer Relevancy</div>
                <div class="metric-val">{metrics_data['answer_relevancy']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Context Recall</div>
                <div class="metric-val">{metrics_data['context_recall']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Context Precision</div>
                <div class="metric-val">{metrics_data['context_precision']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Avg Latency</div>
                <div class="metric-val">{metrics_data['avg_latency_s']:.2f}s</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Draw Plotly Chart
        st.markdown("### Metric Visualizations")
        df_chart = pd.DataFrame({
            "Metric": ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision"],
            "Score": [metrics_data['faithfulness'], metrics_data['answer_relevancy'], metrics_data['context_recall'], metrics_data['context_precision']]
        })
        fig = px.bar(df_chart, x="Metric", y="Score", text="Score", color="Metric", 
                     color_discrete_sequence=["#0D9488", "#3B82F6", "#F59E0B", "#10B981"],
                     range_y=[0, 1.05])
        fig.update_layout(
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            font_color="#F1F5F9",
            xaxis_title="",
            yaxis_title="Ragas Metric Score"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Load detailed results table
        st.markdown("### Query-by-Query Evaluation Breakdown")
        try:
            resp_detail = requests.get(f"{API_URL}/results")
            if resp_detail.status_code == 200 and resp_detail.json().get("status") == "success":
                df_results = pd.DataFrame(resp_detail.json()["data"])
                
                # Format dataframe for display
                df_display = df_results[[
                    "question", "ground_truth", "answer", 
                    "faithfulness", "answer_relevancy", "context_recall", "context_precision", "latency_seconds"
                ]].rename(columns={
                    "question": "Question",
                    "ground_truth": "Ground Truth",
                    "answer": "Generated Answer",
                    "faithfulness": "Faithfulness",
                    "answer_relevancy": "Answer Relevancy",
                    "context_recall": "Context Recall",
                    "context_precision": "Context Precision",
                    "latency_seconds": "Latency (s)"
                })
                st.dataframe(df_display, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load detailed results: {e}")
            
    else:
        st.info("No evaluation runs found. Click 'Run Ragas Evaluation Dataset' to perform benchmark validation.")
