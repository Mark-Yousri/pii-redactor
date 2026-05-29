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


# ── Card / document detection ─────────────────────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _find_document_corners(image: np.ndarray) -> np.ndarray | None:
    """
    Find the 4 corners of a card/document in the image using edge detection.
    Returns (4,2) float32 array in original pixel coords, or None.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for cnt in contours:
        if cv2.contourArea(cnt) < w * h * 0.05:
            continue
        peri = cv2.arcLength(cnt, True)
        for eps_factor in [0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
    return None


def _warp_card(image: np.ndarray, corners: np.ndarray, target_width: int = 2000) -> tuple:
    """
    Perspective-warp the card region to a flat, upscaled image.
    Returns (warped_rgb, M_inv, warp_scale):
      - M_inv: maps warped pixel coords → original image coords
      - warp_scale: pixels in warped space per pixel in natural card space
    """
    rect = _order_points(corners)
    tl, tr, br, bl = rect

    card_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    card_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    card_w = max(card_w, 1)
    card_h = max(card_h, 1)

    dst_nat = np.array([[0, 0], [card_w - 1, 0],
                         [card_w - 1, card_h - 1], [0, card_h - 1]], dtype=np.float32)
    M_fwd = cv2.getPerspectiveTransform(rect, dst_nat)
    M_inv = cv2.getPerspectiveTransform(dst_nat, rect)

    warp_scale = target_width / card_w
    out_w = target_width
    out_h = int(card_h * warp_scale)

    warped_nat = cv2.warpPerspective(image, M_fwd, (card_w, card_h))
    warped = cv2.resize(warped_nat, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    # Enhance contrast
    lab = cv2.cvtColor(warped, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    warped = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)

    return warped, M_inv, warp_scale


def _map_bbox_to_original(left, top, width, height,
                           warp_scale: float, M_inv: np.ndarray) -> tuple:
    """Map a Tesseract bbox from warped+scaled space back to original image pixels."""
    # Step 1: undo the resize scale → natural card coords
    pts = np.array([
        [left / warp_scale,           top / warp_scale],
        [(left + width) / warp_scale, top / warp_scale],
        [(left + width) / warp_scale, (top + height) / warp_scale],
        [left / warp_scale,           (top + height) / warp_scale],
    ], dtype=np.float32).reshape(-1, 1, 2)
    # Step 2: apply inverse perspective transform → original image coords
    orig = cv2.perspectiveTransform(pts, M_inv).reshape(-1, 2)
    x0 = float(np.min(orig[:, 0]))
    y0 = float(np.min(orig[:, 1]))
    x1 = float(np.max(orig[:, 0]))
    y1 = float(np.max(orig[:, 1]))
    return x0, y0, x1, y1


# ── OCR parsing helpers ───────────────────────────────────────────────────────

def _parse_word_tokens(data: dict, page_num: int, pdf_scale: float,
                        img_scale: float = 1.0, M_inv: np.ndarray = None) -> list:
    """Convert Tesseract word-level data to TextToken list."""
    tokens = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text or int(data["conf"][i]) < 20:
            continue
        left, top, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

        if M_inv is not None:
            x0, y0, x1, y1 = _map_bbox_to_original(left, top, w, h, img_scale, M_inv)
        else:
            x0 = left / img_scale
            y0 = top / img_scale
            x1 = (left + w) / img_scale
            y1 = (top + h) / img_scale

        tokens.append(TextToken(
            text=text,
            bbox=(x0 * pdf_scale, y0 * pdf_scale, x1 * pdf_scale, y1 * pdf_scale),
            page_num=page_num,
            source="ocr",
        ))
    return tokens


def _parse_line_tokens(data: dict, page_num: int, pdf_scale: float,
                        img_scale: float = 1.0, M_inv: np.ndarray = None) -> list:
    """Emit line-level bboxes for any OCR line containing Arabic characters."""
    lines: dict[tuple, dict] = {}
    for i, text in enumerate(data["text"]):
        if int(data["conf"][i]) < 10:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        rec = lines.setdefault(key, {"texts": [], "left": [], "top": [], "right": [], "bottom": []})
        rec["texts"].append(text)
        l = data["left"][i]
        t = data["top"][i]
        rec["left"].append(l)
        rec["top"].append(t)
        rec["right"].append(l + data["width"][i])
        rec["bottom"].append(t + data["height"][i])

    tokens = []
    for rec in lines.values():
        combined = " ".join(t for t in rec["texts"] if t.strip())
        if not combined.strip() or not _ARABIC_CHAR.search(combined):
            continue

        left = min(rec["left"])
        top = min(rec["top"])
        right = max(rec["right"])
        bottom = max(rec["bottom"])
        w = right - left
        h = bottom - top

        if M_inv is not None:
            x0, y0, x1, y1 = _map_bbox_to_original(left, top, w, h, img_scale, M_inv)
        else:
            x0 = left / img_scale
            y0 = top / img_scale
            x1 = right / img_scale
            y1 = bottom / img_scale

        tokens.append(TextToken(
            text=combined,
            bbox=(x0 * pdf_scale, y0 * pdf_scale, x1 * pdf_scale, y1 * pdf_scale),
            page_num=page_num,
            source="ocr",
        ))
    return tokens


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
    OCR an image and return TextToken list.

    Strategy:
    1. Try to detect the document/card corners and perspective-warp it to 2000px wide.
       This gives far higher effective resolution on card text in a photo.
    2. Fall back to full-image OCR with CLAHE upscaling if card detection fails.
    """
    lang = _ocr_lang()
    pdf_scale = 72.0 / dpi

    corners = _find_document_corners(image)
    if corners is not None:
        warped, M_inv, warp_scale = _warp_card(image, corners, target_width=2000)
        data = pytesseract.image_to_data(
            warped, lang=lang, config="--psm 6 --oem 1",
            output_type=pytesseract.Output.DICT,
        )
        word_toks = _parse_word_tokens(data, page_num, pdf_scale, warp_scale, M_inv)
        line_toks = _parse_line_tokens(data, page_num, pdf_scale, warp_scale, M_inv)
        if len(word_toks) >= 3:
            return word_toks + line_toks

    # Fallback: upscale full image + CLAHE
    h, w = image.shape[:2]
    scale = max(1.0, 2400 / min(h, w))
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    resized = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)

    data = pytesseract.image_to_data(
        resized, lang=lang, config="--psm 3 --oem 1",
        output_type=pytesseract.Output.DICT,
    )
    word_toks = _parse_word_tokens(data, page_num, pdf_scale, scale)
    line_toks = _parse_line_tokens(data, page_num, pdf_scale, scale)
    return word_toks + line_toks


def extract_tokens(pdf_path: str) -> list:
    if is_born_digital(pdf_path):
        return extract_tokens_digital(pdf_path)

    tokens = []
    for page_num in range(get_page_count(pdf_path)):
        image = rasterize_page(pdf_path, page_num, dpi=200)
        tokens.extend(extract_tokens_ocr(image, page_num, dpi=200))
    return tokens
