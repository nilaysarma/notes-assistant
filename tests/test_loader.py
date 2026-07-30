from pathlib import Path

import pytest

from app.models.page import Page
from app.rag.loader import load_pdf

TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def loaded_pages() -> list[Page]:
    """Load the sample PDF once per test."""

    pdf_path = TEST_DATA_DIR / "sample.pdf"
    return load_pdf(pdf_path)


def test_load_pdf_returns_pages(loaded_pages: list[Page]) -> None:
    assert len(loaded_pages) > 0
    assert all(isinstance(page, Page) for page in loaded_pages)


def test_page_metadata(loaded_pages: list[Page]) -> None:
    first_page = loaded_pages[0]

    # print(first_page)

    assert first_page.page_number == 1
    assert first_page.source == "sample.pdf"


def test_page_contains_text(loaded_pages: list[Page]) -> None:
    assert loaded_pages[0].text.strip()

# from pathlib import Path

# import pytest

# from app.models.page import Page
# from app.rag.loader import load_pdf

# TEST_DATA_DIR = Path(__file__).parent / "data"


# def test_load_pdf_returns_pages() -> None:
#     pdf_path = TEST_DATA_DIR / "sample.pdf"

#     pages = load_pdf(pdf_path)

#     assert len(pages) > 0
#     assert all(isinstance(page, Page) for page in pages)


# def test_page_metadata() -> None:
#     pdf_path = TEST_DATA_DIR / "sample.pdf"

#     pages = load_pdf(pdf_path)

#     first_page = pages[0]

#     assert first_page.page_number == 1
#     assert first_page.source == "sample.pdf"


# def test_page_contains_text() -> None:
#     pdf_path = TEST_DATA_DIR / "sample.pdf"

#     pages = load_pdf(pdf_path)

#     assert pages[0].text.strip()


# def test_missing_pdf() -> None:
#     pdf_path = TEST_DATA_DIR / "missing.pdf"

#     with pytest.raises(FileNotFoundError):
#         load_pdf(pdf_path)