from sentence_transformers import SentenceTransformer

from app.models.chunk import Chunk

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384

_model: SentenceTransformer | None = None


def _load_model() -> SentenceTransformer:
    """Load the embedding model."""

    global _model

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)

    return _model


def _extract_texts(
    chunks: list[Chunk],
) -> list[str]:
    """Extract text from chunks."""

    return [chunk.text for chunk in chunks]


def _embed_texts(
    texts: list[str],
) -> list[list[float]]:
    """Generate normalized embeddings for a list of texts."""

    if not texts:
        return []

    model = _load_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embeddings.tolist()


def embed_chunks(
    chunks: list[Chunk],
) -> list[list[float]]:
    """Generate embeddings for document chunks."""

    texts = _extract_texts(chunks)

    return _embed_texts(texts)


def embed_query(
    query: str,
) -> list[float]:
    """Generate an embedding for a user query."""

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    return _embed_texts([query])[0]
