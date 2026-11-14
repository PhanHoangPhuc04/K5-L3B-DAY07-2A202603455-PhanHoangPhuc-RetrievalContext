"""Local embedding model via sentence-transformers. No mock fallback: errors are raised."""
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "intfloat/multilingual-e5-base"


class E5Embedder:
    """multilingual-e5 expects prefixes: 'query: ' for questions, 'passage: ' for documents.

    Vectors are L2-normalized so cosine distance in the store equals 1 - dot product.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dimensions = int(self.model.get_sentence_embedding_dimension())
        self.max_seq_length = int(self.model.max_seq_length)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode([f"passage: {t}" for t in texts], batch_size=self.batch_size,
                                 normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, question: str) -> list[float]:
        return self.model.encode(f"query: {question}", normalize_embeddings=True).tolist()

    def token_count(self, text: str) -> int:
        return len(self.model.tokenizer(f"passage: {text}")["input_ids"])
