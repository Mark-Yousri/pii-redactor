import fitz
import numpy as np


def is_born_digital(pdf_path: str) -> bool:
    doc = fitz.open(pdf_path)
    char_count = 0
    for page in doc[: min(3, len(doc))]:
        char_count += len(page.get_text())
        if char_count > 10:
            doc.close()
            return True
    doc.close()
    return False


def rasterize_page(pdf_path: str, page_num: int, dpi: int = 200) -> np.ndarray:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


def get_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count
