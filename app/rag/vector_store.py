from pathlib import Path
from typing import cast

import chromadb
from chromadb import Collection
from chromadb.api import ClientAPI
from chromadb.api.types import Embedding, Metadata

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


def _create_chunk(
    chunk_id: str,
    text: str,
    metadata: Metadata,
) -> Chunk:
    """Create a Chunk from ChromaDB search results."""

    source = metadata.get("source")
    page_number = metadata.get("page_number")
    start_char = metadata.get("start_char")
    end_char = metadata.get("end_char")

    if not isinstance(source, str):
        raise ValueError("Invalid chunk source.")

    if not isinstance(page_number, int):
        raise ValueError("Invalid page number.")

    if not isinstance(start_char, int):
        raise ValueError("Invalid start character.")

    if not isinstance(end_char, int):
        raise ValueError("Invalid end character.")

    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page_number=page_number,
        start_char=start_char,
        end_char=end_char,
    )

def search(
    query_embedding: list[float],
    n_results: int = 5,
) -> list[Chunk]:
    """Search for similar chunks."""

    collection = _get_or_create_collection()

    results = collection.query(
        query_embeddings=cast(
            list[Embedding],
            [query_embedding],
        ),
        n_results=n_results,
    )

    if (
        results["documents"] is None
        or results["metadatas"] is None
    ):
        return []

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not ids:
        return []

    return [
        _create_chunk(
            chunk_id=chunk_id,
            text=document,
            metadata=metadata,
        )
        for chunk_id, document, metadata in zip(
            ids,
            documents,
            metadatas,
            strict=True,
        )
    ]

def clear_collection() -> None:
    """Remove all indexed chunks from the collection."""

    client = _get_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        # Collection does not exist.
        pass

    _get_or_create_collection()

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
        ids=[
            chunk.chunk_id
            for chunk in chunks
        ],
        documents=[
            chunk.text
            for chunk in chunks
        ],
        embeddings=cast(
            list[Embedding],
            embeddings,
        ),
        metadatas=[
            _create_metadata(chunk)
            for chunk in chunks
        ],
    )
