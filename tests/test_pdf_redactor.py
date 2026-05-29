"""
Phase 4 acceptance tests — true redaction verification.
Run: python -m pytest tests/test_pdf_redactor.py -v
"""
import os
import fitz
import pytest

from backend.detect.pii_classifier import ClassifiedToken
from backend.detect.text_extractor import TextToken
from backend.redact.pdf_redactor import redact_pdf_digital


SECRET = "SECRETNAME"
OUTPUT = os.path.join("sample_docs", "redacted_out.pdf")


def _make_pdf_with_secret(path: str):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), f"Invoice for {SECRET} dated 2024-01-01")
    doc.save(path)
    doc.close()


@pytest.fixture(scope="module")
def digital_pdf_with_secret(tmp_path_factory):
    p = tmp_path_factory.mktemp("pdfs") / "secret.pdf"
    _make_pdf_with_secret(str(p))
    return str(p)


def _extract_all_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def test_redacted_text_not_in_output(digital_pdf_with_secret, tmp_path):
    # Confirm secret is in original
    assert SECRET in _extract_all_text(digital_pdf_with_secret)

    # Build a classified token that covers the secret word
    doc = fitz.open(digital_pdf_with_secret)
    words = doc[0].get_text("words")
    doc.close()
    secret_word = next(w for w in words if w[4] == SECRET)
    token = TextToken(
        text=SECRET,
        bbox=(secret_word[0], secret_word[1], secret_word[2], secret_word[3]),
        page_num=0,
        source="digital",
    )
    ct = ClassifiedToken(token=token, label="NAME")

    out = str(tmp_path / "out.pdf")
    redact_pdf_digital(
        pdf_path=digital_pdf_with_secret,
        classified_tokens=[ct],
        face_detections=[],
        enabled_types=["name"],
        output_path=out,
    )

    extracted = _extract_all_text(out)
    assert SECRET not in extracted, f"Secret '{SECRET}' still found in redacted output!"


def test_original_not_overwritten(digital_pdf_with_secret, tmp_path):
    out = str(tmp_path / "out.pdf")
    token = TextToken(text="x", bbox=(0, 0, 1, 1), page_num=0, source="digital")
    ct = ClassifiedToken(token=token, label="NAME")
    redact_pdf_digital(digital_pdf_with_secret, [ct], [], ["name"], out)
    # Original still contains the secret
    assert SECRET in _extract_all_text(digital_pdf_with_secret)
