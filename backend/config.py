OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:1.7b"          # fallback: "gemma3:1b"
FACE_DET_THRESHOLD = 0.5
FACE_MODEL_NAME = "buffalo_sc"        # lightweight InsightFace model
MAX_FILE_SIZE_MB = 50
TEMP_DIR = "/tmp/pii_redactor"
PII_TYPES = ["face", "name", "id_number", "date", "address", "signature"]
