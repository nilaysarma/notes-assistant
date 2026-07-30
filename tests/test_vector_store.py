from pathlib import Path

import pytest

from app.models.chunk import Chunk
from app.models.page import Page
from app.rag.embeddings import embed_chunks, embed_query
from app.rag.loader import load_pdf
from app.rag.splitter import split_pages
from app.rag.vector_store import (
    CHROMA_PATH,
    COLLECTION_NAME,
    _get_client,
    _get_or_create_collection,
    add_chunks,
    clear_collection,
    search,
)

TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def loaded_pages() -> list[Page]:
    return load_pdf(TEST_DATA_DIR / "sample.pdf")


@pytest.fixture(scope="module")
def chunks(
    loaded_pages: list[Page],
) -> list[Chunk]:
    return split_pages(loaded_pages)


@pytest.fixture(scope="module")
def embeddings(
    chunks: list[Chunk],
) -> list[list[float]]:
    return embed_chunks(chunks)

def test_chroma_directory_exists() -> None:
    _get_client()

    assert CHROMA_PATH.exists()
    assert CHROMA_PATH.is_dir()


def test_get_or_create_collection_returns_collection() -> None:
    collection = _get_or_create_collection()

    assert collection.name == COLLECTION_NAME


def test_get_or_create_collection_returns_same_collection() -> None:
    collection1 = _get_or_create_collection()
    collection2 = _get_or_create_collection()

    assert collection1.name == COLLECTION_NAME
    assert collection2.name == COLLECTION_NAME


def test_clear_collection_on_empty_database() -> None:
    clear_collection()


def test_clear_collection_removes_all_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    clear_collection()

    add_chunks(
        chunks,
        embeddings,
    )

    clear_collection()

    results = search(
        embed_query("machine learning"),
    )

    assert results == []
