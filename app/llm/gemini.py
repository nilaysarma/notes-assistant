from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai

MODEL_NAME = "gemini-3.5-flash"

load_dotenv()


def _load_client() -> genai.Client:
    """Create a Gemini client."""

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key is None or not api_key.strip():
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set."
        )

    return genai.Client(api_key=api_key)


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

    response = client.models.generate_content(
        model=MODEL_NAME,
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