import json
import re
from dataclasses import dataclass

import httpx

import backend.config as config
from backend.detect.text_extractor import TextToken

# Unicode directional / zero-width marks that Tesseract appends to Arabic tokens.
# These break regex matching on Arabic Unicode ranges.
_CTRL_RE = re.compile(r"[​-‏‪-‮⁦-⁩﻿­]+")


def _clean(text: str) -> str:
    """Strip invisible Unicode control characters, then strip whitespace."""
    return _CTRL_RE.sub("", text).strip()

PATTERNS = {
    "ID_NUMBER": [
        r"\b\d{14}\b",                          # Egyptian NID (Western digits)
        r"[٠-٩]{14}",                 # Egyptian NID (Arabic-Indic digits)
        r"\b[A-Z]{1,2}\d{6,9}\b",              # Passport-style
        r"\b\d{9}\b",                           # Generic 9-digit
        r"[٠-٩\d]{7,9}",             # Mixed/short passport numbers
    ],
    "DATE": [
        r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b",
        r"\b\d{4}[\/\-]\d{2}[\/\-]\d{2}\b",
        r"[٠-٩]{4}[\/\-][٠-٩]{2}[\/\-][٠-٩]{2}",
        r"[٠-٩]{4}[\/\-][٠-٩]{1,2}",  # partial Arabic date (٢٠٠١/٠٧)
        r"\b(0[1-9]|[12]\d|3[01])\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\b",
    ],
}

# Arabic Unicode block: U+0600–U+06FF (covers Arabic letters, including reversed OCR output)
_ARABIC_RE = re.compile(r"^[؀-ۿݐ-ݿࢠ-ࣿ]+$")

# Known Arabic NID field labels — these are structural words, not PII
_ARABIC_LABELS = {
    "الاسم", "القومي", "الرقم", "الميلاد", "تاريخ", "العنوان",
    "الجنس", "الجنسية", "محافظة", "الديانة", "المهنة",
    # reversed forms Tesseract may produce
    # reversed forms Tesseract may produce
    "مسلاا", "يمقلا", "مقرلا", "داليملا", "خيرات", "ناونعلا",
    # Egyptian NID government header — not PII
    "جمهورية", "مصر", "العربية", "بطاقة", "تحقيق", "الشخصية",
    "المصرية", "عربية", "جمهوريه",
    # reversed header words Tesseract may produce
    "ةيروهمج", "رصم", "ةيبرعلا", "ةقاطب", "قيقحت", "ةيصخشلا",
}


def _contains_arabic(s: str) -> bool:
    return bool(re.search(r"[؀-ۿ]", s))


def _is_header_only(t: str) -> bool:
    """True if every Arabic word in the token is a known non-PII label/header."""
    arabic_words = [_clean(w) for w in t.split() if _contains_arabic(w)]
    return all(w in _ARABIC_LABELS for w in arabic_words) if arabic_words else False


def tier1_classify(token: str) -> str | None:
    # Strip Unicode directional marks that Tesseract appends to Arabic tokens
    t = _clean(token)
    if not t:
        return None

    # Numeric patterns first — Arabic-Indic digits sit inside Arabic Unicode range
    # so they must be checked before the Arabic-name heuristic below
    for label, pats in PATTERNS.items():
        for pat in pats:
            if re.fullmatch(pat, t):
                return label

    # Line-level token (contains spaces + Arabic) — skip if it's all header words
    if " " in t and _contains_arabic(t):
        if _is_header_only(t):
            return None
        return "NAME"  # Arabic line with at least one non-label word → PII

    # Single pure Arabic-script token that isn't a field label → NAME
    if _ARABIC_RE.match(t) and t not in _ARABIC_LABELS and len(t) >= 2:
        return "NAME"

    return None


async def tier2_classify_batch(tokens: list[str]) -> list[str]:
    prompt = (
        "You are a PII classifier. Tokens may be in Arabic or English.\n"
        "Classify each token as exactly one of: NAME, ADDRESS, OTHER.\n"
        "NAME includes Arabic and English personal names (e.g. محمد، علي، Hassan).\n"
        "ADDRESS includes cities, streets, governorates in any language.\n"
        "Respond ONLY with a JSON array of labels in the same order as the input.\n"
        "No explanation. No markdown. No extra text.\n\n"
        f"Tokens: {json.dumps(tokens, ensure_ascii=False)}\n"
        "Response:"
    )
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False, "think": False},
            )
            r.raise_for_status()
            raw = r.json().get("response", "")
            # Strip <think>...</think> blocks emitted by reasoning models
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            # Extract first JSON array in the response
            m = re.search(r"\[.*?\]", raw, re.DOTALL)
            if not m:
                raise ValueError("No JSON array in response")
            labels = json.loads(m.group())
            if isinstance(labels, list) and len(labels) == len(tokens):
                return [str(l).upper() for l in labels]
    except Exception:
        pass
    return ["OTHER"] * len(tokens)


@dataclass
class ClassifiedToken:
    token: TextToken
    label: str  # NAME | ID_NUMBER | DATE | ADDRESS | OTHER


async def classify_tokens(tokens: list[TextToken]) -> list[ClassifiedToken]:
    results: list[ClassifiedToken | None] = [None] * len(tokens)
    tier2_indices = []

    for i, t in enumerate(tokens):
        label = tier1_classify(t.text)
        if label:
            results[i] = ClassifiedToken(token=t, label=label)
        else:
            tier2_indices.append(i)

    batch_size = 20
    for start in range(0, len(tier2_indices), batch_size):
        batch_idx = tier2_indices[start : start + batch_size]
        batch_texts = [tokens[i].text for i in batch_idx]
        labels = await tier2_classify_batch(batch_texts)
        for idx, label in zip(batch_idx, labels):
            results[idx] = ClassifiedToken(token=tokens[idx], label=label)

    return results
