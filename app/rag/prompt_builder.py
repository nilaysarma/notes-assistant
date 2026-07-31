from app.models.chunk import Chunk


def _format_context(
    chunks: list[Chunk],
) -> str:
    """Format retrieved chunks for the LLM prompt."""

    sections = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        sections.append(
            f"""[Chunk {index}]

{chunk.text}"""
        )

    return "\n\n--------------------\n\n".join(
        sections
    )


def build_prompt(
    question: str,
    chunks: list[Chunk],
) -> str:
    """Build the prompt sent to the LLM."""

    context = _format_context(
        chunks,
    )

    return f"""You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer cannot be found in the context,
say you don't know.

--------------------

Context

{context}

--------------------

Question

{question}

--------------------

Answer
"""