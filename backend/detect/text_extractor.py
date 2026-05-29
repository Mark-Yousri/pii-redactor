from dataclasses import dataclass

import numpy as np
import pytesseract

import fitz
from backend.utils.pdf_utils import is_born_digital, rasterize_page, get_page_count


@dataclass
class TextToken:
    text: str
    bbox: tuple  # x0, y0, x1, y1 in PDF points (top-left origin)
    page_num: int
    source: str  # "digital" or "ocr"


def extract_tokens_digital(pdf_path: str) -> list:
    tokens = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        for word in page.get_text("words"):
            # word = (x0, y0, x1, y1, word_text, block_no, line_no, word_no)
            text = word[4].strip()
            if text:
                tokens.append(TextToken(
                    text=text,
                    bbox=(word[0], word[1], word[2], word[3]),
                    page_num=page_num,
                    source="digital",
                ))
    doc.close()
    return tokens


def extract_tokens_ocr(image: np.ndarray, page_num: int, dpi: int = 200) -> list:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    tokens = []
    scale = 72.0 / dpi  # convert pixel coords → PDF points
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        conf = int(data["conf"][i])
        if conf < 40:
            continue
        x = data["left"][i] * scale
        y = data["top"][i] * scale
        w = data["width"][i] * scale
        h = data["height"][i] * scale
        tokens.append(TextToken(
            text=text,
            bbox=(x, y, x + w, y + h),
            page_num=page_num,
            source="ocr",
        ))
    return tokens


def extract_tokens(pdf_path: str) -> list:
    if is_born_digital(pdf_path):
        return extract_tokens_digital(pdf_path)

    tokens = []
    for page_num in range(get_page_count(pdf_path)):
        image = rasterize_page(pdf_path, page_num, dpi=200)
        tokens.extend(extract_tokens_ocr(image, page_num, dpi=200))
    return tokens
