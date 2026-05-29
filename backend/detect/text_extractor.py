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


# ── Card / document crop ──────────────────────────────────────────────────────

def _find_card_bbox(image: np.ndarray) -> tuple | None:
    """
    Locate the card/document inside the image.

    Egyptian NIDs are often photographed on a green background. The card
    itself has a beige/cream data area PLUS a darker brown/gold arabesque
    header area. The green mask only catches the lighter data area, so we
    extend the detected region upward by 60% of its height to recover the
    header + name fields.

    Returns (x, y, w, h) in original pixels or None.
    """
    h, w = image.shape[:2]
    min_area = w * h * 0.04

    # ── Strategy A: green background removal ─────────────────────────────────
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Tight green mask: must be clearly green-hued and saturated.
    # Avoids eating the card's brownish-gold arabesque (H~15-35, S low-mid).
    green = cv2.inRange(hsv, np.array([38, 50, 30]), np.array([92, 255, 255]))
    # Very dark pixels (ceiling, deep shadows)
    dark  = cv2.inRange(hsv, np.array([0,  0,  0]),  np.array([180, 255, 45]))
    bg    = cv2.bitwise_or(green, dark)
    card_mask = cv2.bitwise_not(bg)

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (8,  8))
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, k_close)
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_OPEN,  k_open)

    cnts, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) >= min_area:
            x, y, cw, ch = cv2.boundingRect(largest)

            # The green mask only finds the lighter card area (data fields).
            # Extend upward by 60% of detected height to capture the darker
            # arabesque header + name fields that sit above the data section.
            extend_up = int(ch * 0.60)
            y_new  = max(0, y - extend_up)
            ch_new = min(h - y_new, ch + extend_up)

            # Clamp width with a small padding
            pad = 6
            x_new  = max(0, x - pad)
            cw_new = min(w - x_new, cw + 2 * pad)

            return x_new, y_new, cw_new, ch_new

    # ── Strategy B: morphological fallback ───────────────────────────────────
    gray    = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edged   = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
    k_dil   = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edged, k_dil, iterations=4)

    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) >= min_area:
            x, y, cw, ch = cv2.boundingRect(largest)
            return x, y, cw, ch

    return None


def _crop_and_enhance(image: np.ndarray, bbox: tuple, target_width: int = 2000) -> tuple:
    """
    Crop the card region, upscale to target_width, apply CLAHE.
    Returns (enhanced_card, crop_x, crop_y, crop_scale).
    crop_scale converts cropped-and-upscaled pixel → original pixel.
    """
    x, y, cw, ch = bbox
    card = image[y:y+ch, x:x+cw]

    scale = target_width / max(cw, 1)
    card = cv2.resize(card, (target_width, int(ch * scale)), interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(card, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    card = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)

    return card, x, y, scale


# ── OCR parsing ───────────────────────────────────────────────────────────────

def _make_tokens(data: dict, page_num: int, pdf_scale: float,
                 ocr_scale: float, crop_x: int = 0, crop_y: int = 0) -> list:
    """
    Convert Tesseract output to TextToken list (word-level + Arabic line-level).

    Coordinate chain:
      Tesseract px → ÷ ocr_scale → original crop px → + crop offset → original image px
      → × pdf_scale → PDF points
    """
    _ctrl = re.compile(r"[​-‏‪-‮⁦-⁩﻿­]+")
    word_tokens = []
    for i, text in enumerate(data["text"]):
        text = _ctrl.sub("", text).strip()
        if not text or int(data["conf"][i]) < 20:
            continue
        left = data["left"][i];  top = data["top"][i]
        right = left + data["width"][i];  bot = top + data["height"][i]

        x0 = (left  / ocr_scale + crop_x) * pdf_scale
        y0 = (top   / ocr_scale + crop_y) * pdf_scale
        x1 = (right / ocr_scale + crop_x) * pdf_scale
        y1 = (bot   / ocr_scale + crop_y) * pdf_scale
        word_tokens.append(TextToken(text=text, bbox=(x0, y0, x1, y1),
                                     page_num=page_num, source="ocr"))

    # Line-level bboxes for Arabic text
    _ctrl = re.compile(r"[​-‏‪-‮⁦-⁩﻿­]+")
    lines: dict[tuple, dict] = {}
    for i, text in enumerate(data["text"]):
        if int(data["conf"][i]) < 10:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        r = lines.setdefault(key, {"texts": [], "L": [], "T": [], "R": [], "B": []})
        r["texts"].append(_ctrl.sub("", text))
        l = data["left"][i];  t = data["top"][i]
        r["L"].append(l);  r["T"].append(t)
        r["R"].append(l + data["width"][i]);  r["B"].append(t + data["height"][i])

    line_tokens = []
    for r in lines.values():
        combined = " ".join(t for t in r["texts"] if t.strip())
        if not combined.strip() or not _ARABIC_CHAR.search(combined):
            continue
        x0 = (min(r["L"]) / ocr_scale + crop_x) * pdf_scale
        y0 = (min(r["T"]) / ocr_scale + crop_y) * pdf_scale
        x1 = (max(r["R"]) / ocr_scale + crop_x) * pdf_scale
        y1 = (max(r["B"]) / ocr_scale + crop_y) * pdf_scale
        line_tokens.append(TextToken(text=combined, bbox=(x0, y0, x1, y1),
                                     page_num=page_num, source="ocr"))

    return word_tokens + line_tokens


# ── Public API ────────────────────────────────────────────────────────────────

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


def extract_tokens_ocr(image: np.ndarray, page_num: int, dpi: int = 200) -> list:
    """
    OCR pipeline with automatic card crop.

    1. Detect the card/document bounding box (color seg or morphological).
    2. Crop + upscale card to 2000px wide — gives 3-4× effective resolution.
    3. OCR on the high-res card crop.
    4. Map all bboxes back to original image coordinates.
    5. Fall back to full-image OCR at 2400px if card detection fails.
    """
    lang     = _ocr_lang()
    pdf_scale = 72.0 / dpi

    bbox = _find_card_bbox(image)
    if bbox is not None:
        card, cx, cy, card_scale = _crop_and_enhance(image, bbox, target_width=2000)
        data = pytesseract.image_to_data(
            card, lang=lang, config="--psm 6 --oem 1",
            output_type=pytesseract.Output.DICT,
        )
        tokens = _make_tokens(data, page_num, pdf_scale, card_scale, cx, cy)
        if len(tokens) >= 3:
            return tokens

    # Fallback: upscale full image
    h, w = image.shape[:2]
    scale = max(1.0, 2400 / min(h, w))
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    resized = cv2.cvtColor(cv2.merge([clahe.apply(l_ch), a_ch, b_ch]), cv2.COLOR_LAB2RGB)
    data = pytesseract.image_to_data(
        resized, lang=lang, config="--psm 3 --oem 1",
        output_type=pytesseract.Output.DICT,
    )
    return _make_tokens(data, page_num, pdf_scale, scale)


def extract_tokens(pdf_path: str) -> list:
    if is_born_digital(pdf_path):
        return extract_tokens_digital(pdf_path)

    tokens = []
    for page_num in range(get_page_count(pdf_path)):
        image = rasterize_page(pdf_path, page_num, dpi=200)
        tokens.extend(extract_tokens_ocr(image, page_num, dpi=200))
    return tokens
