import asyncio
import base64
import os
import time
import uuid

import fitz
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import backend.config as config
from backend.detect.face import detect_faces
from backend.detect.pii_classifier import classify_tokens
from backend.detect.text_extractor import extract_tokens
from backend.redact.image_redactor import redact_image, redact_scanned_pdf
from backend.redact.pdf_redactor import redact_pdf_digital
from backend.utils.file_utils import cleanup, make_temp_dir
from backend.utils.pdf_utils import (
    get_page_count,
    is_born_digital,
    rasterize_page,
)

app = FastAPI(title="PII Redactor")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store: file_id → {"path": str, "expires": float}
_file_store: dict[str, dict] = {}
_TTL = 600  # 10 minutes


async def _check_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


def _purge_expired():
    now = time.time()
    expired = [k for k, v in _file_store.items() if v["expires"] < now]
    for k in expired:
        cleanup(os.path.dirname(_file_store[k]["path"]))
        del _file_store[k]


def _page_preview_b64(pdf_path: str, page_num: int, dpi: int = 150) -> str:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()
    return base64.b64encode(pix.tobytes("png")).decode()


@app.get("/healthz")
async def healthz():
    ollama_ok = await _check_ollama()
    return {"status": "ok", "ollama": ollama_ok}


@app.post("/redact")
async def redact_endpoint(
    file: UploadFile = File(...),
    enabled_types: str = Form(default="face,name,id_number,date,address"),
    redact_mode: str = Form(default="blur"),
):
    _purge_expired()

    # --- Validate ---
    allowed_ext = {"pdf", "jpg", "jpeg", "png"}
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=422, detail=f"Unsupported file type: .{ext}")

    content = await file.read()
    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {config.MAX_FILE_SIZE_MB} MB limit")

    enabled = [t.strip() for t in enabled_types.split(",") if t.strip()]

    # --- Save upload to temp ---
    tmp_dir = make_temp_dir()
    input_path = os.path.join(tmp_dir, f"input.{ext}")
    output_path = os.path.join(tmp_dir, "output.pdf")
    with open(input_path, "wb") as f:
        f.write(content)

    summary = {"faces": 0, "names": 0, "id_numbers": 0, "dates": 0, "addresses": 0}

    try:
        if ext == "pdf":
            # Text extraction + PII classification
            tokens = extract_tokens(input_path)
            classified = await classify_tokens(tokens)

            # Count
            for ct in classified:
                if ct.label == "NAME" and "name" in enabled:
                    summary["names"] += 1
                elif ct.label == "ID_NUMBER" and "id_number" in enabled:
                    summary["id_numbers"] += 1
                elif ct.label == "DATE" and "date" in enabled:
                    summary["dates"] += 1
                elif ct.label == "ADDRESS" and "address" in enabled:
                    summary["addresses"] += 1

            # Face detection per page
            face_detections = []
            if "face" in enabled:
                for page_num in range(get_page_count(input_path)):
                    img = rasterize_page(input_path, page_num, dpi=150)
                    faces = detect_faces(img, page_num)
                    face_detections.extend(faces)
                summary["faces"] = len(face_detections)

            # Redact
            if is_born_digital(input_path):
                redact_pdf_digital(input_path, classified, face_detections, enabled, output_path)
            else:
                redact_scanned_pdf(input_path, classified, face_detections, enabled, output_path)

        else:
            # Image — face detection only for now
            from PIL import Image as PILImage
            import io
            pil_img = PILImage.open(io.BytesIO(content)).convert("RGB")
            img_arr = np.array(pil_img)

            face_detections = []
            if "face" in enabled:
                face_detections = detect_faces(img_arr, page_num=0)
                summary["faces"] = len(face_detections)

            boxes = [fd.bbox for fd in face_detections]
            redacted_arr = redact_image(img_arr, boxes, mode=redact_mode)

            # Wrap image in a single-page PDF for uniform download
            h, w = redacted_arr.shape[:2]
            doc = fitz.open()
            page = doc.new_page(width=w, height=h)
            pix = fitz.Pixmap(fitz.csRGB, w, h, redacted_arr.tobytes(), False)
            page.insert_image(page.rect, pixmap=pix)
            doc.save(output_path, garbage=4, deflate=True)
            doc.close()

        # --- Generate previews ---
        page_previews = []
        n_pages = get_page_count(output_path)
        for pn in range(n_pages):
            page_previews.append(_page_preview_b64(output_path, pn))

        # --- Verification: check redacted strings are gone ---
        redacted_strings = []
        for ct in (classified if ext == "pdf" else []):
            lbl = ct.label
            tp = {"NAME": "name", "ID_NUMBER": "id_number", "DATE": "date", "ADDRESS": "address"}.get(lbl)
            if tp and tp in enabled:
                redacted_strings.append(ct.token.text)

        verification = _verify_redaction(output_path, redacted_strings)

        # --- Store output ---
        file_id = str(uuid.uuid4())
        _file_store[file_id] = {"path": output_path, "expires": time.time() + _TTL}

        return {
            "output_file_id": file_id,
            "summary": summary,
            "page_previews": page_previews,
            "verification": verification,
        }

    except HTTPException:
        cleanup(tmp_dir)
        raise
    except Exception as exc:
        cleanup(tmp_dir)
        raise HTTPException(status_code=500, detail=str(exc))


def _verify_redaction(output_path: str, redacted_strings: list[str]) -> dict:
    try:
        doc = fitz.open(output_path)
        extracted = "".join(page.get_text() for page in doc)
        doc.close()
        leaked = [s for s in redacted_strings if s and s in extracted]
        return {
            "redacted_strings_found_in_output": leaked,
            "text_extraction_attempted": True,
            "verdict": "LEAK_DETECTED" if leaked else "CLEAN",
        }
    except Exception:
        return {
            "redacted_strings_found_in_output": [],
            "text_extraction_attempted": False,
            "verdict": "UNKNOWN",
        }


@app.get("/download/{file_id}")
async def download(file_id: str):
    entry = _file_store.get(file_id)
    if not entry or not os.path.exists(entry["path"]):
        raise HTTPException(status_code=404, detail="File not found or expired")
    return FileResponse(entry["path"], media_type="application/pdf", filename="redacted.pdf")
