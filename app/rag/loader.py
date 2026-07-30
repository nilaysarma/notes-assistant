from pathlib import Path
from typing import Final, cast

import fitz

from app.models.page import Page
from app.utils.logger import logger

TEXT_EXTRACTION_MODE: Final[str] = "text"


def extract_text(page: fitz.Page) -> str:
    """Extract plain text from a PDF page."""

    return cast(str, page.get_text(TEXT_EXTRACTION_MODE)).strip()


def extract_page(
    page: fitz.Page,
    page_number: int,
    source: str,
) -> Page:
    """Convert a PyMuPDF page into a Page model."""

    return Page(
        page_number=page_number + 1,
        text=extract_text(page),
        source=source,
    )


def load_pdf(pdf_path: Path) -> list[Page]:
    """Load a PDF and return a list of extracted pages."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[Page] = []

    try:
        with fitz.open(pdf_path) as document:
            for page_number in range(document.page_count):
                page = document.load_page(page_number)

                extracted_page = extract_page(
                    page=page,
                    page_number=page_number,
                    source=pdf_path.name,
                )

                if not extracted_page.text:
                    continue

                pages.append(extracted_page)

    except Exception:
        logger.exception("Failed to load PDF: %s", pdf_path)
        raise

    logger.info(
        "Loaded %d pages from %s",
        len(pages),
        pdf_path.name,
    )

    return pages
