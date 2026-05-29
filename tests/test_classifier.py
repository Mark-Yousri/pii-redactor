"""
Phase 3 acceptance tests.
Run: python -m pytest tests/test_classifier.py -v
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from backend.detect.pii_classifier import tier1_classify, classify_tokens, tier2_classify_batch
from backend.detect.text_extractor import TextToken


def tok(text):
    return TextToken(text=text, bbox=(0, 0, 10, 10), page_num=0, source="digital")


# --- tier1 ---

def test_short_number_not_matched():
    assert tier1_classify("29051990") is None


def test_14_digit_is_id():
    assert tier1_classify("29051990123456") == "ID_NUMBER"


def test_date_slash():
    assert tier1_classify("01/06/1990") == "DATE"


def test_date_dash():
    assert tier1_classify("2023-12-31") == "DATE"


def test_no_match_returns_none():
    assert tier1_classify("invoice") is None


# --- tier2 batch count ---

@pytest.mark.asyncio
async def test_40_tokens_makes_exactly_2_ollama_calls():
    tokens = [tok(f"Token{i}") for i in range(40)]
    call_count = 0

    async def fake_batch(texts):
        nonlocal call_count
        call_count += 1
        return ["OTHER"] * len(texts)

    with patch("backend.detect.pii_classifier.tier2_classify_batch", side_effect=fake_batch):
        await classify_tokens(tokens)

    assert call_count == 2, f"Expected 2 calls, got {call_count}"


# --- integration: tier1 tokens don't reach Ollama ---

@pytest.mark.asyncio
async def test_tier1_tokens_skip_ollama():
    tokens = [tok("29051990123456"), tok("01/06/1990")]
    with patch("backend.detect.pii_classifier.tier2_classify_batch") as mock:
        mock.side_effect = AssertionError("Should not call Ollama for tier1 tokens")
        result = await classify_tokens(tokens)

    labels = {ct.token.text: ct.label for ct in result}
    assert labels["29051990123456"] == "ID_NUMBER"
    assert labels["01/06/1990"] == "DATE"
