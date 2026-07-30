from pathlib import Path
from typing import cast

import chromadb
from chromadb import Collection
from chromadb.api import ClientAPI
from chromadb.api.types import Embedding

from app.models.chunk import Chunk

DATA_DIR = Path("data")
CHROMA_PATH = DATA_DIR / "chroma"

COLLECTION_NAME = "notes"

_client: ClientAPI | None = None


def _get_client() -> ClientAPI:
    """Return the ChromaDB client."""

    global _client

    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
        )

    return _client


def _get_or_create_collection() -> Collection:
    """Return the notes collection."""

    client = _get_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
    )


def _create_metadata(
    chunk: Chunk,
) -> dict[str, str | int]:
    """Create metadata for a document chunk."""

    return {
        "source": chunk.source,
        "page_number": chunk.page_number,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }


def add_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    """Add document chunks to ChromaDB."""

    if not chunks:
        return

    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks must match the number of embeddings."
        )

    collection = _get_or_create_collection()

    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=cast(list[Embedding], embeddings),
        metadatas=[
            _create_metadata(chunk)
            for chunk in chunks
        ],
    )
