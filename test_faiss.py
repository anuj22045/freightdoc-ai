
from modules.pdf_processor import extract_text_from_pdf
from modules.text_chunker import create_text_chunks
from modules.embeddings import EmbeddingGenerator
from modules.vector_store import (
    create_vector_store,
    search_similar_chunks
)


PDF_PATH = "sample_documents/bill-of-lading.pdf"


# STEP 1: Extract PDF text
print("\nSTEP 1: Extracting PDF...")

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


# STEP 3: Load embedding model
print("\nSTEP 3: Loading embedding model...")

embedding_generator = EmbeddingGenerator()


# STEP 4: Create FAISS vector store
print("\nSTEP 4: Creating FAISS vector store...")

vector_store = create_vector_store(
    chunks,
    embedding_generator
)

print("FAISS vector store created successfully!")
print("Total vectors stored:", vector_store.index.ntotal)


# STEP 5: Perform similarity search
print("\nSTEP 5: Searching relevant chunks...")

query = "What is the gross weight of the shipment?"

results = search_similar_chunks(
    vector_store,
    query,
    k=2
)


# STEP 6: Display retrieved results
print("\nRETRIEVED DOCUMENT CHUNKS")
print("--------------------------------")

for index, (document, score) in enumerate(results, start=1):

    print(f"\nResult {index}")
    print("Distance Score:", score)
    print("Source:", document.metadata["source"])
    print("Page:", document.metadata["page_number"])
    print("Content:")
    print(document.page_content)


print("\nFAISS TEST COMPLETED SUCCESSFULLY!")
