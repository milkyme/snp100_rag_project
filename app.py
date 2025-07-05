import streamlit as st
import importlib
import reranker as rag
import json

importlib.reload(rag)

st.set_page_config(page_title="10-K RAG Demo", page_icon="📄", layout="wide")

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("🔧 Parameters")
rag.K1 = st.sidebar.slider("FAISS Top-K1", 10, 300, rag.K1, step=10)
rag.K2 = st.sidebar.slider("Rerank Top-K2", 10, min(rag.K1, 100), min(rag.K2, rag.K1), step=5)
rag.M = st.sidebar.slider("Final Top-M", 3, 20, rag.M)

# ── Main ───────────────────────────────────────────────────────────────────
st.title("📑 SEC 10-K RAG Assistant")

# Query input
query = st.text_area("Enter your question", height=100, label_visibility="collapsed",
                     placeholder="e.g., What is Apple's main revenue stream? Compare Microsoft and Google's cloud strategies.")
st.caption("💡 Tip: The system automatically detects companies mentioned in your query and prioritizes relevant documents from those companies.")

col1, col2 = st.columns([1, 4])
with col1:
    search_button = st.button("🔍 Answer", type="primary", use_container_width=True)

if search_button and query.strip():
    # Status and progress bar
    status_container = st.container()
    progress_bar = st.progress(0)
    
    # Retrieve and answer
    with st.spinner("Processing..."):
        # Define a callback function to update status and progress
        def status_callback(msg, progress=None, status_type="info"):
            if progress is not None:
                progress_bar.progress(progress)
            
            if status_type == "success":
                status_container.success(msg)
            elif status_type == "warning":
                status_container.warning(msg)
            else:
                status_container.info(msg)
        
        # Generate answer(All state updates are handled in retrieve and answer functions)
        ans_md = rag.answer(query.strip(), status_callback)
        
        # Complete
        progress_bar.progress(100)
        status_container.success("✅ Analysis complete!")
    
    # Show answer
    st.markdown("### 📋 Answer")
    st.markdown(ans_md)
    
    # Hide status and progress bar
    progress_bar.empty()

# ── Company List by Sector ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🏢 Available Companies by Sector")

# Generate Container for each sector
sector_display_names = {
    "Technology": "💻 Technology",
    "Financial": "🏦 Financial",
    "Healthcare": "🏥 Healthcare", 
    "Consumer": "🛒 Consumer",
    "TelecomMedia": "📡 Telecom & Media",
    "Industrial": "🏭 Industrial",
    "Energy": "⚡ Energy",
    "Utilities": "🔌 Utilities",
    "Others": "📊 Others"
}

tab_names = [sector_display_names.get(sector, sector) for sector in rag.company_by_sector.keys()]
tabs = st.tabs(tab_names)

for idx, (sector, tickers) in enumerate(rag.company_by_sector.items()):
    with tabs[idx]:
        # Show company names from tickers
        companies = []
        for ticker in tickers:
            if ticker in rag.company_names:
                companies.append(f"• [{ticker}] {rag.company_names[ticker]} ")
        
        # 회사 리스트 표시
        # Show list of companies
        for company in companies:
            st.write(company)

