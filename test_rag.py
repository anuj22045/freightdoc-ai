
from modules.pdf_processor import extract_text_from_pdf
from modules.text_chunker import create_text_chunks
from modules.embeddings import EmbeddingGenerator
from modules.vector_store import create_vector_store
from modules.rag_pipeline import answer_question


PDF_PATH = "sample_documents/bill-of-lading.pdf"


# STEP 1: Process PDF
print("\nSTEP 1: Processing PDF...")

pdf_data = extract_text_from_pdf(PDF_PATH)

print("PDF:", pdf_data["filename"]) 


# STEP 2: Create chunks
print("\nSTEP 2: Creating chunks...")

chunks = create_text_chunks(
    pdf_data,
    chunk_size=800,
    chunk_overlap=120
)

print("Total chunks:", len(chunks))


# STEP 3: Generate embeddings and FAISS index
print("\nSTEP 3: Creating embeddings and FAISS...")

embedding_generator = EmbeddingGenerator()

vector_store = create_vector_store(
    chunks,
    embedding_generator
)

print("FAISS vectors:", vector_store.index.ntotal)


# STEP 4: Ask questions
questions = [
    "What is the total gross weight of the shipment?",
    "Who is the consignee and what is the port of discharge?",
    "What is the insurance policy number?"
]


# STEP 5: Generate RAG answers
for question in questions:

    print("\n" + "=" * 60)
    print("QUESTION:", question)

    result = answer_question(
        vector_store,
        question,
        k=3
    )

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")

    for source in result["sources"]:

        print(
            f"- {source['source']} | "
            f"Page {source['page_number']} | "
            f"Distance: {source['distance_score']:.4f}"
        )

print("\nRAG TEST COMPLETED!")
