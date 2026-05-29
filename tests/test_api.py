"""
Phase 5 acceptance tests — API endpoint.
Run: python -m pytest tests/test_api.py -v
"""
import io
import os

import fitz
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def _make_pdf_bytes(text: str = "John Smith ID 29051990123456") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _make_png_bytes() -> bytes:
    import numpy as np
    from PIL import Image
    arr = (np.ones((100, 100, 3)) * 200).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_unsupported_file_type():
    r = client.post("/redact", files={"file": ("test.txt", b"hello", "text/plain")})
    assert r.status_code == 422


def test_file_too_large():
    big = b"x" * (51 * 1024 * 1024)
    r = client.post("/redact", files={"file": ("big.pdf", big, "application/pdf")})
    assert r.status_code == 413


def test_pdf_redact_returns_summary_and_previews():
    pdf = _make_pdf_bytes()
    r = client.post(
        "/redact",
        files={"file": ("doc.pdf", pdf, "application/pdf")},
        data={"enabled_types": "name,id_number,date"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "summary" in body
    assert "page_previews" in body
    assert len(body["page_previews"]) >= 1
    assert "output_file_id" in body


def test_download_redacted_pdf():
    pdf = _make_pdf_bytes()
    r = client.post(
        "/redact",
        files={"file": ("doc.pdf", pdf, "application/pdf")},
        data={"enabled_types": "name"},
    )
    file_id = r.json()["output_file_id"]
    dl = client.get(f"/download/{file_id}")
    assert dl.status_code == 200
    assert dl.headers["content-type"] == "application/pdf"


def test_image_upload_returns_face_summary():
    png = _make_png_bytes()
    r = client.post(
        "/redact",
        files={"file": ("img.png", png, "image/png")},
        data={"enabled_types": "face"},
    )
    assert r.status_code == 200
    assert "summary" in r.json()


def test_download_nonexistent_returns_404():
    r = client.get("/download/does-not-exist")
    assert r.status_code == 404
