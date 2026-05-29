import os
import shutil
import uuid
import backend.config as config


def make_temp_dir() -> str:
    path = os.path.join(config.TEMP_DIR, str(uuid.uuid4()))
    os.makedirs(path, exist_ok=True)
    return path


def cleanup(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
