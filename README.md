# PII Redactor

A fully local web application that detects and redacts PII from images and PDFs — faces, names, ID numbers, dates, and addresses. For born-digital PDFs it performs **true redaction** using PyMuPDF: the underlying text objects are removed, not painted over, so the data is genuinely unrecoverable.

No data leaves your machine.

---

## Why true redaction matters

A black rectangle drawn over text in a PDF is cosmetic. The text layer still exists underneath and can be copied, searched, or extracted with any PDF reader. PyMuPDF's `apply_redactions()` removes the text objects from the document structure entirely. This application verifies this after every redaction and reports a `CLEAN` / `LEAK_DETECTED` verdict.

---

## Features

| Capability | Detail |
|---|---|
| **Face detection** | MediaPipe BlazeFace full-range model — works on small NID card photos |
| **Arabic + English OCR** | Tesseract 5 with `ara+eng` tessdata; card-crop pipeline for phone photos |
| **PII classification** | Tier 1: regex (Egyptian NID 14-digit, Arabic-Indic numerals, dates, passport); Tier 2: local Ollama LLM for names/addresses |
| **True PDF redaction** | PyMuPDF `add_redact_annot` + `apply_redactions()` — text physically removed |
| **Image redaction** | Gaussian blur or solid box over detected regions |
| **Verification badge** | Re-extracts text from output and confirms redacted strings are gone |
| **No Node.js** | Single-file vanilla HTML/JS frontend served directly by FastAPI |

---

## Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| PDF engine | PyMuPDF (fitz) |
| OCR | Tesseract 5 (`ara+eng`) via pytesseract |
| Face detection | MediaPipe BlazeFace full-range |
| LLM classification | Ollama (local) — tested with `gemma4:e2b` |
| Image processing | OpenCV, Pillow |
| Frontend | Vanilla HTML/JS + Tailwind CSS CDN |

---

## Setup

### Prerequisites

- Python 3.11
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or any virtualenv
- [Ollama](https://ollama.com) running locally
- [Tesseract 5](https://github.com/UB-Mannheim/tesseract/wiki) — Windows installer at `C:\Program Files\Tesseract-OCR\`

### 1 — Create conda environment

```bash
conda create -n pii-redactor python=3.11 -y
conda activate pii-redactor
pip install -r backend/requirements.txt
```

### 2 — Install Arabic Tesseract language data

Download `ara.traineddata` from [tessdata_best](https://github.com/tesseract-ocr/tessdata_best) into your Tesseract `tessdata` folder:

```
C:\Program Files\Tesseract-OCR\tessdata\ara.traineddata
```

Or run this once inside the conda env:

```python
import urllib.request
urllib.request.urlretrieve(
    "https://github.com/tesseract-ocr/tessdata_best/raw/main/ara.traineddata",
    r"C:\Program Files\Tesseract-OCR\tessdata\ara.traineddata"
)
```

### 3 — Pull an Ollama model

```bash
ollama pull gemma4:e2b
# or any model you have; update OLLAMA_MODEL in backend/config.py
```

### 4 — Start the server

Run from the **project root** (`d:\...\Anonymizer`):

```powershell
C:\Users\<you>\miniconda3\envs\pii-redactor\Scripts\uvicorn.exe backend.main:app --reload
```

Open **http://localhost:8000** in your browser.

---

## Project structure

```
Anonymizer/
├── backend/
│   ├── main.py                  # FastAPI app, /redact, /download, /healthz
│   ├── config.py                # Tuneable constants (model, thresholds, paths)
│   ├── detect/
│   │   ├── face.py              # MediaPipe face detection + upscale for small faces
│   │   ├── text_extractor.py    # Card-crop OCR pipeline (color seg + dual PSM)
│   │   └── pii_classifier.py    # Tier-1 regex + Tier-2 Ollama classifier
│   ├── redact/
│   │   ├── pdf_redactor.py      # PyMuPDF true redaction for born-digital PDFs
│   │   └── image_redactor.py    # Pixel blur/box for images and scanned pages
│   ├── utils/
│   │   ├── pdf_utils.py         # PDF type detection, page rasterization
│   │   └── file_utils.py        # Temp directory management
│   ├── static/
│   │   └── index.html           # Single-file frontend (no Node.js required)
│   └── requirements.txt
├── tests/
│   ├── test_pdf_extractor.py
│   ├── test_face_detector.py
│   ├── test_classifier.py
│   ├── test_pdf_redactor.py
│   └── test_api.py
├── sample_docs/                  # Put test files here (gitignored)
├── .env.example
└── .gitignore
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Frontend UI |
| `/healthz` | GET | `{"status":"ok","ollama":true/false}` |
| `/redact` | POST | Multipart: `file`, `enabled_types`, `redact_mode` |
| `/download/{file_id}` | GET | Download redacted PDF (10-min TTL) |
| `/debug/ocr` | POST | Debug: shows card detection bbox + all OCR tokens |

### `/redact` response

```json
{
  "output_file_id": "<uuid>",
  "summary": {
    "faces": 1,
    "names": 3,
    "id_numbers": 1,
    "dates": 1,
    "addresses": 1
  },
  "page_previews": ["<base64_png>"],
  "verification": {
    "redacted_strings_found_in_output": [],
    "text_extraction_attempted": true,
    "verdict": "CLEAN"
  }
}
```

---

## How the OCR pipeline works (Arabic NID cards)

Phone photos of NID cards present a specific challenge: the card occupies ~30% of the frame, making card text too small (~20px) for reliable OCR.

The pipeline:

1. **Card detection** — HSV color segmentation isolates the non-green card region. The detected bbox is extended 60% upward to capture the darker arabesque header area that the green mask misses.
2. **Crop + upscale** — The card is cropped and upscaled to 2000px wide (3–4× effective resolution on text).
3. **CLAHE contrast enhancement** — Improves contrast on low-quality photos.
4. **Dual OCR pass** — Tesseract runs with `--psm 6` (uniform block) and `--psm 11` (sparse text). Results are merged and deduplicated. PSM 11 catches isolated short words like first-name fields that PSM 6 misses.
5. **Line-level bboxes** — For any OCR line containing Arabic characters, a full-width line bbox is emitted in addition to word-level tokens. This ensures coverage even when individual Arabic glyphs are misrecognised.
6. **Classification** — Arabic-script tokens are caught by regex tier 1 (no LLM needed). Egyptian government header words (`جمهورية`, `مصر`, `العربية`, `بطاقة تحقيق الشخصية`) are excluded. All other Arabic text on the card is treated as PII.

---

## Configuration (`backend/config.py`)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `gemma4:e2b` | Any locally pulled Ollama model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `FACE_DET_THRESHOLD` | `0.3` | Lower = more sensitive; catches small NID card faces |
| `MAX_FILE_SIZE_MB` | `50` | Upload size limit |
| `TEMP_DIR` | System temp | Where redacted files are held (auto-cleaned after 10 min) |
| `TESSERACT_CMD` | Auto-detected | Looks for `C:\Program Files\Tesseract-OCR\tesseract.exe` first |

---

## Running tests

```bash
cd Anonymizer
python -m pytest tests/ -v
```

Phase 1 tests auto-generate a synthetic PDF in `sample_docs/` if none exists.

---

## Known limitations

- **Signature detection** — not implemented; signatures are visually complex and would require a dedicated model.
- **Heavily skewed photos** — perspective correction is not applied; very angled card photos may yield poor OCR.
- **Non-green backgrounds** — card detection falls back to morphological edge detection; accuracy is lower without a distinctive background colour.
- **Ollama latency** — the Tier-2 LLM call adds 2–5 seconds per batch of 20 tokens. The app falls back to Tier-1-only if Ollama is unavailable.
- **OCR accuracy on degraded photos** — Tesseract accuracy depends heavily on print quality, lighting, and focus. The dual-PSM pipeline mitigates but does not eliminate misses on very low-quality images.
