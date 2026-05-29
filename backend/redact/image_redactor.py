import cv2
import fitz
import numpy as np

from backend.detect.pii_classifier import ClassifiedToken
from backend.detect.face import FaceDetection
from backend.utils.pdf_utils import rasterize_page, get_page_count


_LABEL_TO_TYPE = {
    "NAME": "name",
    "ID_NUMBER": "id_number",
    "DATE": "date",
    "ADDRESS": "address",
}


def redact_image(image: np.ndarray, boxes: list, mode: str = "blur") -> np.ndarray:
    out = image.copy()
    for x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(out.shape[1], x1), min(out.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            continue
        if mode == "blur":
            roi = out[y0:y1, x0:x1]
            blurred = cv2.GaussianBlur(roi, (51, 51), 30)
            out[y0:y1, x0:x1] = blurred
        else:
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), -1)
    return out


def redact_scanned_pdf(
    pdf_path: str,
    classified_tokens: list,
    face_detections: list,
    enabled_types: list,
    output_path: str,
    dpi: int = 200,
) -> None:
    scale = dpi / 72.0  # PDF points → pixels

    faces_by_page: dict[int, list] = {}
    for fd in face_detections:
        faces_by_page.setdefault(fd.page_num, []).append(fd)

    tokens_by_page: dict[int, list] = {}
    for ct in classified_tokens:
        pii_type = _LABEL_TO_TYPE.get(ct.label)
        if pii_type and pii_type in enabled_types:
            tokens_by_page.setdefault(ct.token.page_num, []).append(ct)

    page_count = get_page_count(pdf_path)
    redacted_images = []

    for page_num in range(page_count):
        img = rasterize_page(pdf_path, page_num, dpi=dpi)
        boxes = []

        for ct in tokens_by_page.get(page_num, []):
            x0, y0, x1, y1 = ct.token.bbox
            boxes.append((x0 * scale, y0 * scale, x1 * scale, y1 * scale))

        for fd in faces_by_page.get(page_num, []):
            if "face" in enabled_types:
                boxes.append(fd.bbox)

        if boxes:
            img = redact_image(img, boxes, mode="blur")

        redacted_images.append(img)

    # Rebuild PDF from redacted page images
    doc = fitz.open()
    for img in redacted_images:
        h, w = img.shape[:2]
        page = doc.new_page(width=w * 72 / dpi, height=h * 72 / dpi)
        pix = fitz.Pixmap(fitz.csRGB, w, h, img.tobytes(), False)
        page.insert_image(page.rect, pixmap=pix)
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
