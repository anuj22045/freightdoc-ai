from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def create_text_chunks(pdf_data, chunk_size=800, chunk_overlap=120):
    """Split extracted PDF pages into smaller chunks.

    Each chunk retains its original page number
    and source filename."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function= len,
        separators=["\n\n", "\n", " ", ""]
    )
    documents=[]
    for page in pdf_data["pages"]:
        page_text  =page["text"]
        if not page_text.strip():
            continue

        page_chunks = splitter.create_documents(
            [page_text],
            metadatas=[
                {
                    "source": pdf_data["filename"],
                    "page_number": page["page_number"]
                }
            ]
        )
        documents.extend(page_chunks)

    if not documents:
        raise ValueError("No text chunks were generated")
    return documents