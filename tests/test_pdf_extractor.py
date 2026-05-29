"""
Phase 1 acceptance tests.
Requires two sample PDFs in sample_docs/:
  - digital.pdf  (born-digital)
  - scanned.pdf  (image-only / scanned)
Run: python -m pytest tests/test_pdf_extractor.py -v
"""
import os
import pytest
import fitz

from backend.utils.pdf_utils import is_born_digital, rasterize_page, get_page_count
from backend.detect.text_extractor import extract_tokens


DIGITAL_PDF = os.path.join("sample_docs", "digital.pdf")
SCANNED_PDF = os.path.join("sample_docs", "scanned.pdf")


def _make_digital_pdf(path: str):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World — born digital PDF test content.")
    doc.save(path)
    doc.close()


@pytest.fixture(scope="module", autouse=True)
def ensure_digital_pdf():
    os.makedirs("sample_docs", exist_ok=True)
    if not os.path.exists(DIGITAL_PDF):
        _make_digital_pdf(DIGITAL_PDF)


def test_is_born_digital_true():
    assert is_born_digital(DIGITAL_PDF) is True


def test_digital_tokens_have_correct_source():
    tokens = extract_tokens(DIGITAL_PDF)
    assert len(tokens) > 0
    assert all(t.source == "digital" for t in tokens)


def test_digital_tokens_have_valid_bboxes():
    tokens = extract_tokens(DIGITAL_PDF)
    for t in tokens:
        x0, y0, x1, y1 = t.bbox
        assert x1 > x0 and y1 > y0, f"Invalid bbox for token '{t.text}'"


def test_rasterize_page_shape():
    arr = rasterize_page(DIGITAL_PDF, 0, dpi=150)
    assert arr.ndim == 3 and arr.shape[2] == 3


def test_get_page_count():
    assert get_page_count(DIGITAL_PDF) == 1
