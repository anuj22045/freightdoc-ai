# FreightDoc AI
### AI-Powered Logistics Document Intelligence System

FreightDoc AI is a GenAI-powered application that helps users understand logistics documents such as Bills of Lading, Commercial Invoices and Packing Lists.

It extracts important shipment information, answers questions about uploaded documents and generates AI-powered summaries using Large Language Models and Retrieval-Augmented Generation (RAG).

## Features

- **PDF Processing:** Upload and extract text from logistics PDF documents.
- **Information Extraction:** Extract shipment references, shipper, consignee, ports, cargo details and weights.
- **Document Q&A (RAG):** Ask questions about uploaded documents and receive context-based answers with source references.
- **AI Summarization:** Generate structured summaries of logistics documents.
- **Interactive Interface:** Simple Streamlit interface for document processing and analysis.

## Tech Stack

- **Programming Language:** Python
- **Frontend:** Streamlit
- **LLM:** Groq API (`openai/gpt-oss-20b`)
- **LLM Framework:** LangChain
- **PDF Processing:** PyPDF
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Store:** FAISS
- **Data Validation:** Pydantic

## Project Structure

```text
FreightDoc-AI/
│
├── modules/
│   ├── pdf_processor.py
│   ├── text_chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   ├── information_extractor.py
│   └── summarizer.py
│
├── sample_documents/
├── app.py
├── config.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/FreightDoc-AI.git
cd FreightDoc-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```powershell
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Groq API Key

1. Visit [Groq Console](https://console.groq.com/keys).
2. Sign in and generate an API key.
3. Create a `.env` file in the project's root directory.
4. Add your credentials:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Replace `your_groq_api_key` with your actual API key.

**Important:** Never share your API key or upload your `.env` file to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## How to Use

1. Upload a logistics PDF document.
2. Click **Process Document** to extract text and create the FAISS vector index.
3. Navigate to **Ask Questions (RAG)** to ask questions about the document.
4. Open **Extract Information** to view structured shipment details and cargo information.
5. Open **Document Summary** to view the AI-generated summary.

## Future Enhancements

- Support for scanned PDFs using OCR.
- Support for multiple document uploads.
- Advanced RAG with hybrid search and reranking.
- Improved document validation and extraction accuracy.
- Integration with additional logistics document formats.

---

**Developed as a Generative AI project to simplify logistics document understanding and information retrieval.**