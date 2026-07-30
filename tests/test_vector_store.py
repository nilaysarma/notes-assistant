from app.rag.vector_store import (
    CHROMA_PATH,
    COLLECTION_NAME,
    _get_client,
    _get_or_create_collection,
)


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