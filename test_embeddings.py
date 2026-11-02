
from modules.pdf_processor import extract_text_from_pdf
from modules.text_chunker import create_text_chunks
from modules.embeddings import EmbeddingGenerator


PDF_PATH = "sample_documents/bill-of-lading.pdf"


# STEP 1: Extract PDF text
print("\nSTEP 1: Extracting PDF text...")

pdf_data = extract_text_from_pdf(PDF_PATH)

print("PDF:", pdf_data["filename"])
print("Pages:", pdf_data["total_pages"])


# STEP 2: Create chunks
print("\nSTEP 2: Creating text chunks...")

chunks = create_text_chunks(
    pdf_data,
    chunk_size=800,
    chunk_overlap=120
)

print("Total chunks:", len(chunks))


# Display chunks
for index, chunk in enumerate(chunks):

    print(f"\n--- CHUNK {index + 1} ---")
    print("Page:", chunk.metadata["page_number"])
    print("Text:", chunk.page_content[:250])


# STEP 3: Generate embeddings
print("\nSTEP 3: Loading embedding model...")

embedding_generator = EmbeddingGenerator()

print("\nSTEP 4: Generating embeddings...")

embeddings = embedding_generator.generate_embeddings(chunks)


# STEP 5: Display results
print("\nEMBEDDING GENERATION SUCCESSFUL")
print("--------------------------------")

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("Vector dimension:", embeddings.shape[1])

print("\nFirst chunk embedding:")
print(embeddings[0][:10])
