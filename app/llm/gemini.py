from __future__ import annotations

from google import genai

from app.core.config import settings
from app.utils.logger import logger


def _load_client() -> genai.Client:
    if not settings.google_api_key.strip():
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set."
        )

    return genai.Client(
        api_key=settings.google_api_key,
)


def _validate_prompt(
    prompt: str,
) -> str:
    """Validate the prompt."""

    prompt = prompt.strip()

    if not prompt:
        raise ValueError(
            "Prompt cannot be empty."
        )

    return prompt


def _generate_response(
    client: genai.Client,
    prompt: str,
) -> str:
    """Generate a response using Gemini."""
    
    logger.info(
        "Generating response with Gemini"
    )

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
    )

    text = response.text

    if text is None or not text.strip():
        raise ValueError(
            "Gemini returned an empty response."
        )

    return text.strip()


def generate(
    prompt: str,
) -> str:
    """Generate text from Gemini."""

    prompt = _validate_prompt(
        prompt,
    )

    client = _load_client()

    return _generate_response(
        client,
        prompt,
    )