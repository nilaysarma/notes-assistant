from pydantic import BaseModel, ConfigDict

from app.models.chunk import Chunk


class AnswerResult(BaseModel):
    """Result returned by the RAG service."""

    model_config = ConfigDict(
        frozen=True,
    )

    answer: str

    chunks: list[Chunk]