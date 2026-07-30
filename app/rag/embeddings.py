from sentence_transformers import SentenceTransformer

from app.models.chunk import Chunk

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384

_model: SentenceTransformer | None = None


def _load_model() -> SentenceTransformer:
    """Load the embedding model if necessary."""

    global _model

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)

    return _model


def _extract_texts(chunks: list[Chunk]) -> list[str]:
    """Extract text from document chunks."""

    return [chunk.text for chunk in chunks]


def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """Generate embeddings for document chunks."""

    if not chunks:
        return []

    texts = _extract_texts(chunks)

    model = _load_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embeddings.tolist()