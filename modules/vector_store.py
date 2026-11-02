from langchain_community.vectorstores import FAISS

def create_vector_store(documents, embedding_generator):
    """
    Create a FAISS vector store from document chunks.
    """

    if not documents:
        raise ValueError("no documents provided")

    vector_store = FAISS.from_documents(
        documents = documents,
        embedding = embedding_generator
    )
    return vector_store

def search_similar_chunks(vector_store, query, k=2):
    """
    Retrieve the most relevant chunks for a query.
    """

    if not query.strip():
        raise ValueError("Search query cannot be empty")

    k = min(k, vector_store.index.ntotal)

    results = vector_store.similarity_search_with_score(
        query, k=k
    )

    return results


def save_vector_store(vector_store, path="faiss_index"):
    """
    Save FAISS index and associated document data locally.
    """

    vector_store.save_local(path)


def load_vector_store(path, embedding_generator):
    """
    Load an existing local FAISS vector store.
    """
    vector_store = FAISS.load_local(
        path, embedding_generator,
        allow_dangerous_deserialization=True
    )
    return vector_store
