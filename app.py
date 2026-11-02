
import streamlit as st
import pandas as pd

from modules.pdf_processor import extract_text_from_pdf
from modules.text_chunker import create_text_chunks
from modules.embeddings import EmbeddingGenerator
from modules.vector_store import create_vector_store
from modules.rag_pipeline import answer_question
from modules.information_extractor import extract_shipment_information
from modules.summarizer import summarize_document


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FreightDoc AI",
    page_icon="🚢",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS - FOREST GREEN + EMERALD + WHITE
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F8FAF9;
}

[data-testid="stSidebar"] {
    background-color: #063D32;
}

[data-testid="stSidebar"] * {
    color: white;
}

[data-testid="stSidebar"] .stRadio label {
    color: white !important;
}

h1, h2, h3 {
    color: #123C32;
}

.stButton > button {
    background-color: #07865B;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 8px 18px;
}

.stButton > button:hover {
    background-color: #056B49;
    color: white;
    border: none;
}

[data-testid="stFileUploader"] {
    background-color: #F0F8F4;
    border: 1px dashed #0D9B72;
    border-radius: 8px;
    padding: 12px;
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #E1EAE5;
    padding: 15px;
    border-radius: 8px;
}

hr {
    border-color: #DDE8E2;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------

if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "extracted_info" not in st.session_state:
    st.session_state.extracted_info = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_file_key" not in st.session_state:
    st.session_state.processed_file_key = None


# --------------------------------------------------
# LOAD EMBEDDING MODEL ONCE
# --------------------------------------------------

@st.cache_resource
def get_embedding_generator():
    return EmbeddingGenerator()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        "<h2 style='text-align:center;color:white;'>🚢 FreightDoc AI</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center;color:#CDE8DC;'>"
        "Logistics Document Intelligence"
        "</p>",
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Upload & Process",
            "Document Overview",
            "Extract Information",
            "Ask Questions (RAG)",
            "Document Summary"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Supported Documents")

    st.markdown("""
    - Bill of Lading (B/L)
    - Commercial Invoice
    - Packing List
    """)

    st.divider()

    st.caption("FreightDoc AI | MVP")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.title("FreightDoc AI")

st.markdown(
    "AI-Powered Logistics Document Intelligence (MVP)"
)

st.divider()


# --------------------------------------------------
# CHECK CURRENT DOCUMENT
# --------------------------------------------------

uploaded_file = None

if page == "Upload & Process":

    st.subheader("Upload a Logistics Document")

    uploaded_file = st.file_uploader(
        "Upload your logistics PDF",
        type=["pdf"],
        help="Upload one PDF at a time."
    )

    if uploaded_file is not None:

        current_file_key = (
            f"{uploaded_file.name}_{uploaded_file.size}"
        )

        if current_file_key != st.session_state.processed_file_key:

            st.info(
                f"Selected document: {uploaded_file.name}"
            )

            if st.button("Process Document", type="primary"):

                try:

                    progress = st.progress(0)

                    status = st.empty()

                    # STEP 1: PDF PROCESSING
                    status.write("Processing PDF...")

                    uploaded_file.seek(0)

                    pdf_data = extract_text_from_pdf(uploaded_file)

                    progress.progress(20)

                    # STEP 2: TEXT CHUNKING
                    status.write("Creating text chunks...")

                    chunks = create_text_chunks(pdf_data)

                    progress.progress(40)

                    # STEP 3: EMBEDDINGS
                    status.write("Generating embeddings...")

                    embedding_generator = get_embedding_generator()

                    progress.progress(65)

                    # STEP 4: FAISS VECTOR STORE
                    status.write("Creating FAISS vector store...")

                    vector_store = create_vector_store(
                        chunks,
                        embedding_generator
                    )

                    progress.progress(100)

                    # SAVE RESULTS IN SESSION STATE
                    st.session_state.pdf_data = pdf_data

                    st.session_state.chunks = chunks

                    st.session_state.vector_store = vector_store

                    st.session_state.extracted_info = None

                    st.session_state.summary = None

                    st.session_state.chat_history = []

                    st.session_state.processed_file_key = current_file_key

                    status.empty()

                    progress.empty()

                    st.success("Document processed successfully!")

                    st.rerun()

                except Exception as e:

                    st.error(f"Document processing failed: {e}")

        else:

            st.success("Document processed successfully!")

    else:

        st.info("Upload a PDF document to get started.")


# --------------------------------------------------
# DOCUMENT STATUS
# --------------------------------------------------

document_ready = (
    st.session_state.pdf_data is not None
    and st.session_state.vector_store is not None
)

if page != "Upload & Process" and not document_ready:

    st.warning(
        "Please upload and process a logistics document first."
    )

    st.stop()


# --------------------------------------------------
# DOCUMENT OVERVIEW
# --------------------------------------------------

elif page == "Document Overview":

    st.subheader("Document Overview")

    pdf_data = st.session_state.pdf_data

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Pages", pdf_data["total_pages"])

    col2.metric("Total Chunks", len(st.session_state.chunks))

    col3.metric(
        "Total Characters",
        f'{pdf_data["total_characters"]:,}'
    )

    st.divider()

    st.markdown("### Document Information")

    st.write("**Filename:**", pdf_data["filename"])

    st.write("**Document Type:** Logistics PDF")

    st.write("**Processing Status:** Successfully processed")

    st.write("**Vector Database:** FAISS")

    st.write("**Embedding Model:** all-MiniLM-L6-v2")


# --------------------------------------------------
# INFORMATION EXTRACTION
# --------------------------------------------------

elif page == "Extract Information":

    st.subheader("Extracted Shipment Information")

    if st.session_state.extracted_info is None:

        if st.button("Extract Information", type="primary"):

            with st.spinner("Extracting shipment information..."):

                try:

                    extracted_info = extract_shipment_information(
                        st.session_state.pdf_data["full_text"]
                    )

                    st.session_state.extracted_info = extracted_info

                    st.rerun()

                except Exception as e:

                    st.error(f"Information extraction failed: {e}")

    if st.session_state.extracted_info is not None:

        data = st.session_state.extracted_info

        st.markdown("### Shipment Details")

        general_fields = {
            key: value
            for key, value in data.items()
            if key != "cargo_items"
            and value is not None
        }

        details_df = pd.DataFrame(
            [
                {
                    "Field": key.replace("_", " ").title(),
                    "Value": value
                }
                for key, value in general_fields.items()
            ]
        )

        st.dataframe(
            details_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Cargo Items")

        cargo_items = data.get("cargo_items", [])

        if cargo_items:

            cargo_df = pd.DataFrame(cargo_items)

            st.dataframe(
                cargo_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No cargo items were found.")


# --------------------------------------------------
# ASK QUESTIONS USING RAG
# --------------------------------------------------

elif page == "Ask Questions (RAG)":

    st.subheader("Ask Questions About Your Document")

    st.write(
        "Ask questions about shipment details, cargo, "
        "weights, ports, and other document information."
    )

    with st.form("question_form", clear_on_submit=True):

        question = st.text_input(
            "Enter your question",
            placeholder="What is the total gross weight?"
        )

        submitted = st.form_submit_button("Ask Question")

    if submitted:

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("Finding the answer..."):

                try:

                    response = answer_question(
                        st.session_state.vector_store,
                        question
                    )

                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": response["answer"],
                            "sources": response["sources"]
                        }
                    )

                except Exception as e:

                    st.error(f"Unable to answer the question: {e}")

    for item in reversed(st.session_state.chat_history):

        st.markdown("**You:**")

        st.write(item["question"])

        st.markdown("**FreightDoc AI:**")

        st.write(item["answer"])

        with st.expander("View Sources"):

            for source in item["sources"]:

                st.markdown(
                    f'**{source["citation"]}** — '
                    f'{source["source"]}, '
                    f'Page {source["page_number"]}'
                )

                st.write(source["content"])


# --------------------------------------------------
# DOCUMENT SUMMARY
# --------------------------------------------------

elif page == "Document Summary":

    st.subheader("AI Document Summary")

    if st.session_state.summary is None:

        if st.button("Generate Summary", type="primary"):

            with st.spinner("Generating document summary..."):

                try:

                    summary = summarize_document(
                        st.session_state.pdf_data["full_text"]
                    )

                    st.session_state.summary = summary

                    st.rerun()

                except Exception as e:

                    st.error(f"Summary generation failed: {e}")

    if st.session_state.summary is not None:

        st.markdown(st.session_state.summary)
