from dataclasses import dataclass
import os

import numpy as np
import pytesseract

import fitz
import backend.config as config
from backend.utils.pdf_utils import is_born_digital, rasterize_page, get_page_count

pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


def _ocr_lang() -> str:
    """Return the best available language string for Tesseract."""
    tessdata = os.path.join(os.path.dirname(config.TESSERACT_CMD), "tessdata")
    has_arabic = os.path.exists(os.path.join(tessdata, "ara.traineddata"))
    return "ara+eng" if has_arabic else "eng"


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


def _preprocess_for_ocr(image: np.ndarray) -> tuple:
    """
    Upscale small images and enhance contrast for better OCR accuracy.
    Returns (processed_image, scale_factor) — bboxes must be divided by scale.
    """
    import cv2
    h, w = image.shape[:2]
    scale = 1.0
    # Upscale if shorter side < 1200px — NID card photos are often small
    if min(h, w) < 1200:
        scale = 1200 / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # CLAHE contrast enhancement helps with low-contrast ID card text
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image, scale


def extract_tokens_ocr(image: np.ndarray, page_num: int, dpi: int = 200) -> list:
    lang = _ocr_lang()
    processed, img_scale = _preprocess_for_ocr(image)
    config_str = "--psm 3 --oem 1"
    data = pytesseract.image_to_data(
        processed,
        lang=lang,
        config=config_str,
        output_type=pytesseract.Output.DICT,
    )
    tokens = []
    pdf_scale = 72.0 / dpi  # pixel coords → PDF points
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        conf = int(data["conf"][i])
        if conf < 25:
            continue
        # Divide by img_scale to get back to original pixel space, then to PDF points
        x = (data["left"][i] / img_scale) * pdf_scale
        y = (data["top"][i] / img_scale) * pdf_scale
        w = (data["width"][i] / img_scale) * pdf_scale
        h = (data["height"][i] / img_scale) * pdf_scale
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
