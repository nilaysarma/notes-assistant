from unittest.mock import Mock

import pytest
from google import genai

from app.llm.gemini import (
    _load_client,
    _validate_prompt,
    generate,
)


def test_validate_prompt() -> None:
    """Test validating a valid prompt."""

    prompt = "Explain machine learning."

    assert _validate_prompt(prompt) == prompt


def test_validate_prompt_strips_whitespace() -> None:
    """Test prompt whitespace is stripped."""

    prompt = "  Explain machine learning.  "

    assert (
        _validate_prompt(prompt)
        == "Explain machine learning."
    )


def test_validate_prompt_empty() -> None:
    """Test empty prompt raises ValueError."""

    with pytest.raises(
        ValueError,
        match="Prompt cannot be empty.",
    ):
        _validate_prompt("")


def test_validate_prompt_whitespace() -> None:
    """Test whitespace-only prompt raises ValueError."""

    with pytest.raises(
        ValueError,
        match="Prompt cannot be empty.",
    ):
        _validate_prompt("     ")


def test_load_client_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test loading client without an API key."""

    monkeypatch.delenv(
        "GEMINI_API_KEY",
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="GEMINI_API_KEY",
    ):
        _load_client()


def test_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test generate orchestrates correctly."""

    client = Mock(spec=genai.Client)

    monkeypatch.setattr(
        "app.llm.gemini._load_client",
        lambda: client,
    )

    generate_response = Mock(
        return_value="Test response",
    )

    monkeypatch.setattr(
        "app.llm.gemini._generate_response",
        generate_response,
    )

    result = generate(
        "  Hello!  "
    )

    assert result == "Test response"

    generate_response.assert_called_once_with(
        client,
        "Hello!",
    )