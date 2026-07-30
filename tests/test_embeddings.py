from pathlib import Path

import pytest

from app.models.chunk import Chunk
from app.models.page import Page
from app.rag.embeddings import (
    EMBEDDING_DIMENSION,
    embed_chunks,
    embed_query,
)
from app.rag.loader import load_pdf
from app.rag.splitter import split_pages

TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def loaded_pages() -> list[Page]:
    """Load the sample PDF."""

    return load_pdf(TEST_DATA_DIR / "sample.pdf")


@pytest.fixture(scope="module")
def chunks(
    loaded_pages: list[Page],
) -> list[Chunk]:
    """Split the sample PDF into chunks."""

    return split_pages(loaded_pages)


@pytest.fixture(scope="module")
def embeddings(
    chunks: list[Chunk],
) -> list[list[float]]:
    """Generate embeddings for the sample PDF."""

    return embed_chunks(chunks)


@pytest.fixture(scope="module")
def query_embedding() -> list[float]:
    """Generate an embedding for a sample query."""

    return embed_query("What is machine learning?")


def test_embed_chunks_returns_embeddings(
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    assert len(embeddings) == len(chunks)


def test_embedding_is_list_of_floats(
    embeddings: list[list[float]],
) -> None:
    first_embedding = embeddings[0]

    assert isinstance(first_embedding, list)
    assert all(
        isinstance(value, float)
        for value in first_embedding
    )


def test_embedding_dimension(
    embeddings: list[list[float]],
) -> None:
    assert len(embeddings[0]) == EMBEDDING_DIMENSION


def test_embeddings_are_normalized(
    embeddings: list[list[float]],
) -> None:
    first_embedding = embeddings[0]

    magnitude = sum(
        value**2
        for value in first_embedding
    ) ** 0.5

    assert magnitude == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_empty_chunks_returns_empty_list() -> None:
    embeddings = embed_chunks([])

    assert embeddings == []


def test_embed_query_returns_embedding(
    query_embedding: list[float],
) -> None:
    assert isinstance(query_embedding, list)
    assert len(query_embedding) == EMBEDDING_DIMENSION


def test_embed_query_returns_floats(
    query_embedding: list[float],
) -> None:
    assert all(
        isinstance(value, float)
        for value in query_embedding
    )


def test_embed_query_returns_normalized_embedding(
    query_embedding: list[float],
) -> None:
    magnitude = sum(
        value**2
        for value in query_embedding
    ) ** 0.5

    assert magnitude == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_embed_query_raises_error_for_empty_query() -> None:
    with pytest.raises(ValueError):
        embed_query("")

# from pathlib import Path

# import pytest

# from app.models.chunk import Chunk
# from app.models.page import Page
# from app.rag.embeddings import (
#     EMBEDDING_DIMENSION,
#     embed_chunks,
# )
# from app.rag.loader import load_pdf
# from app.rag.splitter import split_pages

# TEST_DATA_DIR = Path(__file__).parent / "data"


# @pytest.fixture(scope="module")
# def loaded_pages() -> list[Page]:
#     """Load the sample PDF."""

#     return load_pdf(TEST_DATA_DIR / "sample.pdf")


# @pytest.fixture(scope="module")
# def chunks(
#     loaded_pages: list[Page],
# ) -> list[Chunk]:
#     """Split the sample PDF into chunks."""

#     return split_pages(loaded_pages)


# @pytest.fixture(scope="module")
# def embeddings(
#     chunks: list[Chunk],
# ) -> list[list[float]]:
#     """Generate embeddings for the sample PDF."""

#     return embed_chunks(chunks)


# def test_embed_chunks_returns_embeddings(
#     chunks: list[Chunk],
#     embeddings: list[list[float]],
# ) -> None:
#     assert len(embeddings) == len(chunks)


# def test_embedding_is_list_of_floats(
#     embeddings: list[list[float]],
# ) -> None:
#     first_embedding = embeddings[0]

#     assert isinstance(first_embedding, list)
#     assert all(
#         isinstance(value, float)
#         for value in first_embedding
#     )


# def test_embedding_dimension(
#     embeddings: list[list[float]],
# ) -> None:
#     assert len(embeddings[0]) == EMBEDDING_DIMENSION


# def test_embeddings_are_normalized(
#     embeddings: list[list[float]],
# ) -> None:
#     first_embedding = embeddings[0]

#     magnitude = sum(
#         value**2
#         for value in first_embedding
#     ) ** 0.5

#     assert magnitude == pytest.approx(
#         1.0,
#         abs=1e-6,
#     )


# def test_empty_chunks_returns_empty_list() -> None:
#     embeddings = embed_chunks([])

#     assert embeddings == []

