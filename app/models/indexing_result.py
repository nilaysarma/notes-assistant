from pydantic import BaseModel, ConfigDict


class IndexingResult(BaseModel):
    """Result returned after indexing a document."""

    model_config = ConfigDict(
        frozen=True,
    )

    page_count: int

    chunk_count: int