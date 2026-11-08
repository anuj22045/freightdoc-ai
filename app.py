
import base64
import streamlit as st
import pandas as pd
from pathlib import Path
import textwrap
from modules.pdf_processor import extract_text_from_pdf
from modules.text_chunker import create_text_chunks
from modules.embeddings import EmbeddingGenerator
from modules.vector_store import create_vector_store
from modules.rag_pipeline import answer_question
from modules.information_extractor import extract_shipment_information
from modules.summarizer import summarize_document
import html

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FreightDoc AI",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# HELPER: Load images as base64
# --------------------------------------------------

def load_image_b64(path: str) -> str:
    """Return a base64-encoded data URI for an image file."""
    p = Path(path)
    if not p.exists():
        return ""
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{encoded}"


ASSETS = Path(__file__).parent / "assets"
SHIP_LOGO_B64   = load_image_b64(str(ASSETS / "ship_logo.png"))
PORT_ILLUS_B64  = load_image_b64(str(ASSETS / "port_illustration.png"))


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* ── Global background & Base Typography ───────── */
.stApp {
    background-color: #F4F8F6;
    font-size: 16px;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    font-size: 1.05rem;
    line-height: 1.65;
}

/* ── Sidebar ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #063D32 !important;
    padding-top: 0 !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

/* All sidebar text white */
[data-testid="stSidebar"],
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Radio option styling */
[data-testid="stSidebar"] .stRadio > div {
    gap: 3px;
}

[data-testid="stSidebar"] .stRadio label {
    color: #CBD9D3 !important;
    font-size: 1.02rem;
    padding: 8px 14px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    transition: background 0.15s;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background-color: rgba(255,255,255,0.08) !important;
}

/* Active radio — Streamlit adds aria-checked */
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) {
    background-color: #07865B !important;
    color: white !important;
    font-weight: 600;
}

/* Hide the circular radio dot itself */
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
    margin: 0.6rem 0;
}

/* Sidebar section headings */
[data-testid="stSidebar"] .sidebar-section-heading {
    color: #A8C8BB !important;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 12px;
    margin-top: 4px;
}

/* Sidebar doc list items */
[data-testid="stSidebar"] .sidebar-doc-item {
    color: #CBD9D3 !important;
    font-size: 0.96rem;
    padding: 5px 12px 5px 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Main content header ────────────────────────── */
.main-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0 6px 0;
    border-bottom: 2px solid #0D7A55;
    margin-bottom: 24px;
}

.main-header-left h1 {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #063D32 !important;
    line-height: 1.1 !important;
}

.main-header-left p {
    margin: 2px 0 0 0;
    color: #4A7C66;
    font-size: 1rem;
}

.main-header-right img {
    height: 90px;
    object-fit: contain;
}

/* ── Buttons ─────────────────────────────────────── */
.stButton > button {
    background-color: #07865B !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    font-size: 1.02rem !important;
    transition: background 0.15s !important;
}

.stButton > button:hover {
    background-color: #056B49 !important;
    color: white !important;
}

.stButton > button:active {
    background-color: #045A3D !important;
}

/* ── File uploader ───────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: #EEF7F2 !important;
    border: 2px dashed #0D9B72 !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

[data-testid="stFileUploader"] * {
    font-size: 1rem !important;
}

[data-testid="stFileUploader"] svg {
    color: #07865B !important;
}

/* ── Success / info / warning boxes ─────────────── */
.stAlert {
    border-radius: 8px !important;
}

div[data-testid="stAlert"] * {
    font-size: 1.05rem !important;
}

/* ── Section cards ────────────────────────────────── */
.section-card {
    background: white;
    border: 1px solid #DCE9E3;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 18px;
}

.section-card h3 {
    margin-top: 0 !important;
    color: #063D32 !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px !important;
}

/* ── Metrics ─────────────────────────────────────── */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #DCE9E3;
    border-radius: 8px;
    padding: 14px 18px;
}

[data-testid="stMetricLabel"] {
    color: #4A7C66 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #063D32 !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ── Tables / DataFrames ─────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #DCE9E3 !important;
    border-radius: 8px !important;
}

/* ── Expanders (RAG Sources) ─────────────────────── */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE9E3 !important;
    border-radius: 8px !important;
}

[data-testid="stExpander"] details summary {
    background-color: #EEF7F2 !important;
    color: #063D32 !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 10px 14px !important;
}

[data-testid="stExpander"] details summary p {
    color: #063D32 !important;
    font-size: 1.05rem !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background-color: #FFFFFF !important;
    color: #263B33 !important;
}

[data-testid="stExpander"] [data-testid="stMarkdownContainer"],
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] [data-testid="stText"],
[data-testid="stExpander"] pre {
    color: #263B33 !important;
    background-color: #FFFFFF !important;
    font-size: 1.02rem !important;
}

/* PDF source text inside expander */
.source-content {
    background-color: #FFFFFF !important;
    color: #263B33 !important;
    padding: 16px;
    border-radius: 6px;
    font-size: 15.5px;
    line-height: 1.7;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

/* ── Headings in main area ───────────────────────── */
h2 {
    color: #063D32 !important;
    font-size: 1.65rem !important;
}

h3, [data-testid="stMarkdownContainer"] h3 {
    color: #063D32 !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    margin-top: 18px;
    margin-bottom: 14px;
}

h4 {
    color: #063D32 !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
}

/* ── Progress bar ────────────────────────────────── */
[data-testid="stProgressBar"] > div {
    background-color: #07865B !important;
}

/* ── Text Input (RAG) ────────────────────────────── */
[data-testid="stTextInput"] input {
    font-size: 1.05rem !important;
    padding: 10px 14px !important;
}

[data-testid="stTextInput"] input::placeholder {
    font-size: 1rem !important;
}

/* ── Question answer boxes ────────────────────────── */
.answer-box {
    background: #EEF7F2;
    border-left: 4px solid #07865B;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 14px 0;
    font-size: 1.08rem;
}

.answer-label {
    color: #07865B;
    font-weight: 700;
    font-size: 1.08rem;
    margin-bottom: 8px;
}

/* ── Processing status box ────────────────────────── */
.stat-row {
    display: flex;
    gap: 12px;
    margin-top: 8px;
    flex-wrap: wrap;
}

.stat-chip {
    background: #D4EDDA;
    color: #0A5C3E;
    border-radius: 20px;
    padding: 5px 16px;
    font-size: 0.95rem;
    font-weight: 600;
}

/* ── Summary area ────────────────────────────────── */
.summary-content {
    background: #063d32;
    # background: #dae8e5;
    border: 1px solid #DCE9E3;
    border-radius: 10px;
    padding: 24px 28px;
    line-height: 1.8;
    font-size: 1.08rem;
}

.summary-content p,
.summary-content li {
    font-size: 1.08rem !important;
    line-height: 1.75 !important;
}

.summary-content h1 {
    font-size: 1.6rem !important;
    margin-bottom: 16px !important;
}

.summary-content h2, .summary-content h3 {
    color: #063D32 !important;
    font-size: 1.35rem !important;
    margin-top: 18px !important;
    margin-bottom: 12px !important;
}

.summary-content table {
    font-size: 1.02rem !important;
}

.summary-content table th,
.summary-content table td {
    padding: 10px 14px !important;
}

/* ── Cargo total row ──────────────────────────────── */
.total-row {
    background: #EEF7F2 !important;
    font-weight: 700 !important;
}

/* -----------------------------------------
    EXTRACT INFORMATION — SHIPMENT DETAILS & CARGO TABLES
----------------------------------------- */

/* Shipment Details Table */
.details-table-container {
    width: 100%;
    overflow-x: auto;
    border: 1px solid #DCE9E3;
    border-radius: 8px;
    background-color: #FFFFFF;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    margin-bottom: 20px;
}

.details-table {
    width: 100%;
    border-collapse: collapse;
    background-color: #FFFFFF;
    color: #1A201E;
    font-size: 15.5px;
}

.details-table tr {
    border-bottom: 1px solid #DCE9E3;
}

.details-table tr:last-child {
    border-bottom: none;
}

.details-table th {
    background-color: #EEF7F2;
    color: #063D32;
    font-weight: 600;
    font-size: 15px;
    text-align: left;
    width: 44%;
    padding: 12px 16px;
    border-right: 1px solid #DCE9E3;
    white-space: nowrap;
    vertical-align: middle;
}

.details-table td {
    background-color: #FFFFFF;
    color: #1A201E;
    font-size: 15.5px;
    font-weight: 500;
    padding: 12px 16px;
    word-break: break-word;
    overflow-wrap: anywhere;
    vertical-align: middle;
}

.details-table tr:hover td {
    background-color: #F8FCFA;
}

/* Cargo Items Table */
.cargo-table-container {
    width: 100%;
    overflow-x: auto;
    border: 1px solid #DCE9E3;
    border-radius: 8px;
    background-color: #FFFFFF;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    margin-bottom: 20px;
}

.cargo-table {
    width: 100%;
    border-collapse: collapse;
    background-color: #FFFFFF;
    color: #1A201E;
    font-size: 15.5px;
    min-width: 650px;
}

.cargo-table th {
    background-color: #DDF3E7;
    color: #063D32;
    font-weight: 700;
    font-size: 15px;
    text-align: left;
    padding: 13px 16px;
    border-bottom: 2px solid #C4E2D2;
    border-right: 1px solid #DCE9E3;
    white-space: nowrap;
}

.cargo-table th:last-child {
    border-right: none;
}

.cargo-table td {
    background-color: #FFFFFF;
    color: #1A201E;
    padding: 12px 16px;
    border-bottom: 1px solid #E6EFEA;
    border-right: 1px solid #E6EFEA;
    font-size: 15.5px;
}

.cargo-table td:last-child {
    border-right: none;
}

.cargo-table tbody tr:hover td {
    background-color: #F4FAF6;
}

.cargo-table .total-row td {
    background-color: #D4EDDA !important;
    color: #063D32 !important;
    font-weight: 700 !important;
    border-top: 2px solid #07865B !important;
    border-bottom: none !important;
    font-size: 16px !important;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------

defaults = {
    "pdf_data": None,
    "chunks": None,
    "vector_store": None,
    "extracted_info": None,
    "summary": None,
    "last_question": "",
    "last_answer": None,
    "processed_file_key": None,
    "page": "Upload & Process",
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# --------------------------------------------------
# LOAD EMBEDDING MODEL ONCE (cached)
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedding_generator():
    return EmbeddingGenerator()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    # Logo + brand
    if SHIP_LOGO_B64:
        st.markdown(
            f"""
            <div style="text-align:center; padding: 10px 0 4px 0;">
                <img src="{SHIP_LOGO_B64}"
                    style="height:72px; object-fit:contain;" />
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div style="text-align:center; padding-bottom:6px;">
            <span style="font-size:1.3rem; font-weight:800;
                        color:white; letter-spacing:0.5px;">
                FreightDoc AI
            </span><br>
            <span style="font-size:0.78rem; color:#A8C8BB;">
                Logistics Document Intelligence
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # Navigation
    page = st.radio(
        "Navigation",
        options=[
            "⬆️  Upload & Process",
            "💬  Ask Questions (RAG)",
            "📋  Extract Information",
            "📄  Document Summary",
        ],
        label_visibility="collapsed",
        key="nav_radio"
    )

    # Strip the emoji prefix to get the clean page name
    page_name = page.split("  ", 1)[1] if "  " in page else page

    st.divider()

    # Supported documents
    st.markdown(
        '<div class="sidebar-section-heading">Supported Documents</div>',
        unsafe_allow_html=True
    )

    for icon, label in [
        ("📄", "Bill of Lading (B/L)"),
        ("🧾", "Commercial Invoice"),
        ("📦", "Packing List"),
    ]:
        st.markdown(
            f'<div class="sidebar-doc-item">{icon} {label}</div>',
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown(
        '<p style="color:#6B9E8A; font-size:0.75rem; text-align:center; margin:0;">'
        'FreightDoc AI &nbsp;|&nbsp; MVP'
        '</p>',
        unsafe_allow_html=True
    )


# --------------------------------------------------
# MAIN HEADER (consistent across all pages)
# --------------------------------------------------

port_img_tag = (
    f'<img src="{PORT_ILLUS_B64}" />'
    if PORT_ILLUS_B64 else ""
)

st.markdown(
    f"""
    <div class="main-header">
        <div class="main-header-left">
            <h1>FreightDoc AI</h1>
            <p>AI-Powered Logistics Document Intelligence (MVP)</p>
        </div>
        <div class="main-header-right">
            {port_img_tag}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# INTERFACE 1 — UPLOAD & PROCESS
# --------------------------------------------------

if page_name == "Upload & Process":

    st.markdown(
        "### ⬆️  Upload a Logistics Document",
    )

    uploaded_file = st.file_uploader(
        "Upload your logistics PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload one PDF at a time (max 50 MB)."
    )

    # ── If a file is selected ──────────────────────
    if uploaded_file is not None:

        current_key = f"{uploaded_file.name}_{uploaded_file.size}"
        already_processed = (
            current_key == st.session_state.processed_file_key
        )

        # File info row
        col_name, col_btn = st.columns([3, 1])

        with col_name:
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:10px;
                            padding:8px 14px; background:#FFF;
                            border:1px solid #DCE9E3; border-radius:8px;">
                    <span style="font-size:1.1rem;">📕</span>
                    <span style="font-weight:600; color:#063D32;">
                        {uploaded_file.name}
                    </span>
                    <span style="color:#888; font-size:0.82rem;">
                        ({uploaded_file.size // 1024} KB)
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_btn:
            process_clicked = st.button(
                "▶  Process Document",
                disabled=already_processed,
                use_container_width=True
            )

        # ── Processing logic ───────────────────────
        if process_clicked and not already_processed:

            progress_bar = st.progress(0, text="Starting…")
            status_msg   = st.empty()

            try:
                # Step 1 — PDF extraction
                status_msg.write("📄 Extracting text from PDF…")
                uploaded_file.seek(0)
                pdf_data = extract_text_from_pdf(uploaded_file)
                progress_bar.progress(20, text="PDF extracted")

                # Step 2 — Chunking
                status_msg.write("✂️  Creating text chunks…")
                chunks = create_text_chunks(pdf_data)
                progress_bar.progress(45, text="Chunks created")

                # Step 3 — Embeddings
                status_msg.write("🧠 Generating embeddings…")
                embedding_generator = get_embedding_generator()
                progress_bar.progress(70, text="Embeddings generated")

                # Step 4 — FAISS
                status_msg.write("🗂️  Building FAISS vector store…")
                vector_store = create_vector_store(chunks, embedding_generator)
                progress_bar.progress(100, text="Done!")

                # Persist in session state
                st.session_state.pdf_data         = pdf_data
                st.session_state.chunks           = chunks
                st.session_state.vector_store     = vector_store
                st.session_state.extracted_info   = None
                st.session_state.summary          = None
                st.session_state.last_answer      = None
                st.session_state.last_question    = ""
                st.session_state.processed_file_key = current_key

                status_msg.empty()
                progress_bar.empty()

                st.rerun()

            except Exception as exc:
                progress_bar.empty()
                status_msg.empty()
                st.error(f"❌ Processing failed: {exc}")

        # ── Success / stats panel ──────────────────
        if already_processed and st.session_state.pdf_data:
            pdf = st.session_state.pdf_data
            n_chunks = len(st.session_state.chunks)

            st.success("✅ Document processed successfully!")

            st.markdown(
                f"""
                <div class="stat-row">
                    <span class="stat-chip">📄 Pages: {pdf["total_pages"]}</span>
                    <span class="stat-chip">✂️ Chunks: {n_chunks}</span>
                    <span class="stat-chip">🧠 Embeddings: {n_chunks} (FAISS Index Ready)</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        # Nothing uploaded yet
        if st.session_state.pdf_data is not None:
            pdf = st.session_state.pdf_data
            n_chunks = len(st.session_state.chunks)
            st.success(
                f"✅ **{pdf['filename']}** is loaded and ready.  "
                f"Upload a new file to switch documents."
            )
            st.markdown(
                f"""
                <div class="stat-row">
                    <span class="stat-chip">📄 Pages: {pdf["total_pages"]}</span>
                    <span class="stat-chip">✂️ Chunks: {n_chunks}</span>
                    <span class="stat-chip">🧠 Embeddings: {n_chunks} (FAISS Index Ready)</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Upload a PDF document to get started.")


# --------------------------------------------------
# DOCUMENT READY GUARD (pages 2-4)
# --------------------------------------------------

document_ready = (
    st.session_state.pdf_data is not None
    and st.session_state.vector_store is not None
)

if page_name != "Upload & Process" and not document_ready:
    st.warning(
        "⚠️  Please upload and process a logistics document first.  "
        "Go to **Upload & Process** in the sidebar."
    )
    st.stop()


# --------------------------------------------------
# INTERFACE 2 — ASK QUESTIONS (RAG)
# --------------------------------------------------

elif page_name == "Ask Questions (RAG)":

    st.markdown("### 💬 Ask Questions (RAG)")
    st.markdown(
        "Ask any question about the uploaded document.",
        unsafe_allow_html=False
    )

    # Question input + button on the same row
    q_col, btn_col = st.columns([5, 1])

    with q_col:
        question = st.text_input(
            "Question",
            value=st.session_state.last_question,
            placeholder="What is the total gross weight of the shipment?",
            label_visibility="collapsed"
        )

    with btn_col:
        ask_clicked = st.button("Ask", use_container_width=True)

    # ── Run RAG ───────────────────────────────────
    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Finding the answer…"):
                try:
                    response = answer_question(
                        st.session_state.vector_store,
                        question
                    )
                    st.session_state.last_question = question
                    st.session_state.last_answer   = response
                except Exception as exc:
                    st.error(f"❌ Could not answer the question: {exc}")

    # ── Display answer ─────────────────────────────
    if st.session_state.last_answer:
        resp = st.session_state.last_answer

        st.markdown(
            f"""
            <div class="answer-box">
                <div class="answer-label">Answer:</div>
                <div style="color:#1A1A1A; line-height:1.65;">
                    {resp["answer"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### 🗂️ Sources:")

        for i, src in enumerate(resp["sources"], start=1):
            with st.expander(f"Source {i}  (Page {src['page_number']})"):
                st.markdown(
                    f"**File:** {src['source']}  &nbsp;|&nbsp;  "
                    f"**Page:** {src['page_number']}  &nbsp;|&nbsp;  "
                    f"**FAISS Distance:** {src['distance_score']:.4f}"
                )
                st.markdown("---")
                st.markdown(
    f'<div class="source-content">{html.escape(src["content"])}</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# INTERFACE 3 — EXTRACT INFORMATION
# --------------------------------------------------

elif page_name == "Extract Information":

    st.markdown("### 📋 Extracted Information")

    # Extract information if not already available
    if st.session_state.extracted_info is None:

        if st.button("Extract Information", type="primary"):

            with st.spinner("Extracting shipment information..."):

                try:
                    info = extract_shipment_information(
                        st.session_state.pdf_data["full_text"]
                    )

                    st.session_state.extracted_info = info
                    st.rerun()

                except Exception as exc:
                    st.error(f"Extraction failed: {exc}")

    # Display extracted information
    if st.session_state.extracted_info is not None:

        data = st.session_state.extracted_info

        # --------------------------------------------------
        # SHIPMENT DETAILS
        # --------------------------------------------------

        st.markdown("### 🚢 Shipment Details")

        # Map known fields to preferred display labels and column placement
        # Left column: document references and net weight
        # Right column: parties, dates, ports, and gross weight
        FIELD_MAPPING = [
            ("document_type", "Document Type", "left"),
            ("shipment_id", "Shipment ID", "left"),
            ("bill_of_lading_number", "B/L Number", "left"),
            ("carrier_reference", "Carrier Reference", "left"),
            ("unique_consignment_reference", "Consignment Reference", "left"),
            ("total_net_weight", "Total Net Weight", "left"),
            ("shipper", "Shipper", "right"),
            ("consignee", "Consignee", "right"),
            ("shipment_date", "Shipment Date", "right"),
            ("port_of_loading", "Port of Loading", "right"),
            ("port_of_discharge", "Port of Discharge", "right"),
            ("total_gross_weight", "Total Gross Weight", "right"),
        ]

        extracted = {}
        for k, v in data.items():
            if k == "cargo_items" or v is None:
                continue
            val_str = str(v).strip()
            if not val_str or val_str.lower() == "none":
                continue
            extracted[k] = val_str

        # If shipment_id was not set but shipper_reference was, map it
        if "shipment_id" not in extracted and "shipper_reference" in extracted:
            extracted["shipment_id"] = extracted["shipper_reference"]

        left_items = []
        right_items = []
        handled_keys = set()

        for key, label, side in FIELD_MAPPING:
            if key in extracted:
                if side == "left":
                    left_items.append((label, extracted[key]))
                else:
                    right_items.append((label, extracted[key]))
                handled_keys.add(key)

        # Append any remaining extracted fields without duplicating
        for key, val in extracted.items():
            if key in handled_keys:
                continue
            if key == "shipper_reference" and val == extracted.get("shipment_id"):
                continue
            label = key.replace("_", " ").title()
            if len(left_items) <= len(right_items):
                left_items.append((label, val))
            else:
                right_items.append((label, val))

        def render_detail_table(items):
            rows = []
            for label, value in items:
                safe_label = html.escape(str(label))
                safe_value = html.escape(str(value))
                rows.append(f"<tr><th>{safe_label}</th><td>{safe_value}</td></tr>")
            rows_html = "".join(rows)
            return f'<div class="details-table-container"><table class="details-table"><tbody>{rows_html}</tbody></table></div>'

        if left_items or right_items:
            left_col, right_col = st.columns(2)

            with left_col:
                if left_items:
                    st.markdown(
                        render_detail_table(left_items),
                        unsafe_allow_html=True
                    )

            with right_col:
                if right_items:
                    st.markdown(
                        render_detail_table(right_items),
                        unsafe_allow_html=True
                    )
        else:
            st.info("No shipment details were extracted.")

        # --------------------------------------------------
        # CARGO ITEMS
        # --------------------------------------------------

        cargo_items = data.get("cargo_items", [])
        cargo_count = len(cargo_items) if cargo_items else 0
        header_title = f"### 📦 Cargo Items ({cargo_count})" if cargo_count else "### 📦 Cargo Items"
        st.markdown(header_title)

        if cargo_items:
            thead_html = (
                "<thead><tr>"
                "<th style='width: 6%; text-align: center;'>#</th>"
                "<th>Product Description</th>"
                "<th style='width: 18%;'>Quantity</th>"
                "<th style='width: 18%;'>Net Weight</th>"
                "<th style='width: 18%;'>Gross Weight</th>"
                "</tr></thead>"
            )

            rows = []
            for index, item in enumerate(cargo_items, start=1):
                description = html.escape(str(item.get("product_description") or "—"))
                quantity = html.escape(str(item.get("quantity") or "—"))
                net_weight = html.escape(str(item.get("net_weight") or "—"))
                gross_weight = html.escape(str(item.get("gross_weight") or "—"))
                rows.append(
                    f"<tr>"
                    f"<td style='text-align: center; color: #527A68; font-weight: 600;'>{index}</td>"
                    f"<td style='font-weight: 500;'>{description}</td>"
                    f"<td>{quantity}</td>"
                    f"<td>{net_weight}</td>"
                    f"<td>{gross_weight}</td>"
                    f"</tr>"
                )

            total_net = html.escape(str(data.get("total_net_weight") or "—"))
            total_gross = html.escape(str(data.get("total_gross_weight") or "—"))

            total_row_html = (
                f"<tr class='total-row'>"
                f"<td style='text-align: center;'></td>"
                f"<td>Total</td>"
                f"<td>—</td>"
                f"<td>{total_net}</td>"
                f"<td>{total_gross}</td>"
                f"</tr>"
            )

            tbody_html = f"<tbody>{''.join(rows)}{total_row_html}</tbody>"
            cargo_table_html = (
                f'<div class="cargo-table-container">'
                f'<table class="cargo-table">{thead_html}{tbody_html}</table>'
                f'</div>'
            )

            st.markdown(cargo_table_html, unsafe_allow_html=True)
        else:
            st.info("No individual cargo items were extracted.")

        # --------------------------------------------------
        # RE-EXTRACT BUTTON
        # --------------------------------------------------

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        if st.button("🔄 Re-extract Information"):
            st.session_state.extracted_info = None
            st.rerun()



# --------------------------------------------------
# INTERFACE 4 — DOCUMENT SUMMARY
# --------------------------------------------------

elif page_name == "Document Summary":

    st.markdown("### 📄 Document Summary")

    if st.session_state.summary is None:

        if st.button("Generate Summary", type="primary"):

            with st.spinner("Generating document summary…"):
                try:
                    summary = summarize_document(
                        st.session_state.pdf_data["full_text"]
                    )
                    st.session_state.summary = summary
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Summary generation failed: {exc}")

    if st.session_state.summary is not None:

        st.markdown(
            f'<div class="summary-content">{st.session_state.summary}</div>',
            unsafe_allow_html=True
        )

        st.markdown("&nbsp;")
        if st.button("🔄 Regenerate Summary"):
            st.session_state.summary = None
            st.rerun()
