import fitz

from backend.detect.pii_classifier import ClassifiedToken
from backend.detect.face import FaceDetection
from backend.utils.pdf_utils import rasterize_page


_LABEL_TO_TYPE = {
    "NAME": "name",
    "ID_NUMBER": "id_number",
    "DATE": "date",
    "ADDRESS": "address",
}


def redact_pdf_digital(
    pdf_path: str,
    classified_tokens: list,
    face_detections: list,
    enabled_types: list,
    output_path: str,
) -> None:
    doc = fitz.open(pdf_path)

    # Group face detections by page
    faces_by_page: dict[int, list] = {}
    for fd in face_detections:
        faces_by_page.setdefault(fd.page_num, []).append(fd)

    for page_num, page in enumerate(doc):
        # Add text redaction annotations
        for ct in classified_tokens:
            if ct.token.page_num != page_num:
                continue
            pii_type = _LABEL_TO_TYPE.get(ct.label)
            if pii_type and pii_type in enabled_types:
                rect = fitz.Rect(*ct.token.bbox)
                page.add_redact_annot(rect, fill=(0, 0, 0))

        # Apply text redactions — removes underlying text objects
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        # Pixel-level face redaction (faces have no text layer)
        if "face" in enabled_types and page_num in faces_by_page:
            img = rasterize_page(pdf_path, page_num, dpi=150)
            h_px, w_px = img.shape[:2]
            page_rect = page.rect
            x_scale = page_rect.width / w_px
            y_scale = page_rect.height / h_px
            for fd in faces_by_page[page_num]:
                fx0, fy0, fx1, fy1 = fd.bbox
                pdf_rect = fitz.Rect(fx0 * x_scale, fy0 * y_scale, fx1 * x_scale, fy1 * y_scale)
                page.draw_rect(pdf_rect, color=(0, 0, 0), fill=(0, 0, 0))

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
