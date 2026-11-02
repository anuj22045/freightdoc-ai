
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class EmbeddingGenerator(Embeddings):

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(
            model_name,
            device="cpu"
        )

    # Generate embeddings for multiple documents
    def embed_documents(self, texts):

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings.tolist()

    # Generate embedding for a user query
    def embed_query(self, text):

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.tolist()

