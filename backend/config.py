import os
import sys
import tempfile

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e2b"
FACE_DET_THRESHOLD = 0.3  # lower threshold catches small/angled NID card faces
FACE_MODEL_NAME = "buffalo_sc"
MAX_FILE_SIZE_MB = 50
TEMP_DIR = os.path.join(tempfile.gettempdir(), "pii_redactor")
PII_TYPES = ["face", "name", "id_number", "date", "address", "signature"]

def _find_tesseract() -> str:
    # Standard Windows installer location
    win_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(win_default):
        return win_default
    # Conda env fallback
    conda_prefix = os.environ.get("CONDA_PREFIX") or os.path.dirname(sys.executable)
    candidate = os.path.join(conda_prefix, "Library", "bin", "tesseract.exe")
    if os.path.exists(candidate):
        return candidate
    return "tesseract"

TESSERACT_CMD = _find_tesseract()
