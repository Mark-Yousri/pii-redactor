from dataclasses import dataclass
import os
import re

import cv2
import numpy as np
import pytesseract

import fitz
import backend.config as config
from backend.utils.pdf_utils import is_born_digital, rasterize_page, get_page_count

pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

_ARABIC_CHAR = re.compile(r"[؀-ۿ]")


def _ocr_lang() -> str:
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
    """Upscale and enhance contrast. Returns (processed, scale_factor)."""
    h, w = image.shape[:2]
    scale = 1.0
    if min(h, w) < 1200:
        scale = 1200 / min(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)
    return image, scale


def extract_tokens_ocr(image: np.ndarray, page_num: int, dpi: int = 200) -> list:
    """
    Returns word-level tokens for English/numeric text PLUS line-level bounding
    boxes for any line that contains Arabic characters.

    Line-level bboxes are more robust for Arabic NID data because Tesseract
    often misrecognises individual Arabic glyphs but still localises lines correctly.
    """
    lang = _ocr_lang()
    processed, img_scale = _preprocess_for_ocr(image)
    cfg = "--psm 3 --oem 1"
    data = pytesseract.image_to_data(
        processed, lang=lang, config=cfg, output_type=pytesseract.Output.DICT
    )

    pdf_scale = 72.0 / dpi  # pixel → PDF-point conversion

    # ── Word-level tokens ────────────────────────────────────────────────────
    word_tokens = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text or int(data["conf"][i]) < 25:
            continue
        x = (data["left"][i] / img_scale) * pdf_scale
        y = (data["top"][i] / img_scale) * pdf_scale
        w = (data["width"][i] / img_scale) * pdf_scale
        h = (data["height"][i] / img_scale) * pdf_scale
        word_tokens.append(TextToken(
            text=text,
            bbox=(x, y, x + w, y + h),
            page_num=page_num,
            source="ocr",
        ))

    # ── Line-level tokens for Arabic text ────────────────────────────────────
    # Group words by (block_num, par_num, line_num) to build line bboxes.
    # Any line whose concatenated text contains Arabic characters gets emitted
    # as a single wide token so the whole line is redacted even if individual
    # word recognition is poor.
    lines: dict[tuple, dict] = {}
    for i, text in enumerate(data["text"]):
        if int(data["conf"][i]) < 15:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        rec = lines.setdefault(key, {"texts": [], "left": [], "top": [], "right": [], "bottom": []})
        rec["texts"].append(text)
        left = data["left"][i]
        top = data["top"][i]
        rec["left"].append(left)
        rec["top"].append(top)
        rec["right"].append(left + data["width"][i])
        rec["bottom"].append(top + data["height"][i])

    line_tokens = []
    for key, rec in lines.items():
        combined = " ".join(t for t in rec["texts"] if t.strip())
        if not combined.strip():
            continue
        if not _ARABIC_CHAR.search(combined):
            continue  # English-only lines handled at word level
        x0 = (min(rec["left"]) / img_scale) * pdf_scale
        y0 = (min(rec["top"]) / img_scale) * pdf_scale
        x1 = (max(rec["right"]) / img_scale) * pdf_scale
        y1 = (max(rec["bottom"]) / img_scale) * pdf_scale
        line_tokens.append(TextToken(
            text=combined,
            bbox=(x0, y0, x1, y1),
            page_num=page_num,
            source="ocr",
        ))

    return word_tokens + line_tokens


def extract_tokens(pdf_path: str) -> list:
    if is_born_digital(pdf_path):
        return extract_tokens_digital(pdf_path)

    tokens = []
    for page_num in range(get_page_count(pdf_path)):
        image = rasterize_page(pdf_path, page_num, dpi=200)
        tokens.extend(extract_tokens_ocr(image, page_num, dpi=200))
    return tokens
