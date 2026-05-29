import json
import re
from dataclasses import dataclass

import httpx

import backend.config as config
from backend.detect.text_extractor import TextToken

PATTERNS = {
    "ID_NUMBER": [
        r"\b\d{14}\b",
        r"\b[A-Z]{1,2}\d{6,9}\b",
        r"\b\d{9}\b",
    ],
    "DATE": [
        r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b",
        r"\b\d{4}[\/\-]\d{2}[\/\-]\d{2}\b",
        r"\b(0[1-9]|[12]\d|3[01])\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\b",
    ],
}


def tier1_classify(token: str) -> str | None:
    for label, pats in PATTERNS.items():
        for pat in pats:
            if re.fullmatch(pat, token.strip()):
                return label
    return None


async def tier2_classify_batch(tokens: list[str]) -> list[str]:
    prompt = (
        "You are a PII classifier. Classify each token as exactly one of: NAME, ADDRESS, OTHER.\n"
        "Respond ONLY with a JSON array of labels in the same order as the input.\n"
        "No explanation. No markdown.\n\n"
        f"Tokens: {json.dumps(tokens)}\n"
        "Response:"
    )
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False},
            )
            r.raise_for_status()
            raw = r.json().get("response", "")
            labels = json.loads(raw.strip())
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
