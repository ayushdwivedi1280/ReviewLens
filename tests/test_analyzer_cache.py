"""Tests for ReviewAnalyzer batching, single retry on bad AI output, and SQLite caching."""

import json
import re
from typing import Dict, Any, List
import pytest

from app.config import Settings
from app.llm import LLMClient
from app.cache import Cache
from app.cleaning import CleanedReview
from app.analyzer import ReviewAnalyzer, check_rating_mismatch


class MockLLM(LLMClient):
    """Predictable mock LLM for testing batching, retries, and errors."""

    def __init__(self, override_responses: Dict[int, Any] = None) -> None:
        self.settings = Settings(
            provider="groq",
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
            api_key="mock-key",
            batch_size=5,
        )
        self.call_count = 0
        self.override_responses = override_responses or {}

    def complete(self, system_prompt: str, user_message: str, json_mode: bool = True) -> str:
        self.call_count += 1
        # Extract items from <reviews> tags
        match = re.search(r"<reviews>\s*(\[[\s\S]*?\])\s*</reviews>", user_message)
        items = json.loads(match.group(1)) if match else []

        results = []
        for it in items:
            item_id = it["id"]
            if item_id in self.override_responses:
                results.append(self.override_responses[item_id])
            else:
                results.append({
                    "id": item_id,
                    "sentiment": "positive",
                    "topics": ["quality"],
                    "summary": f"Summary for {it['product']}.",
                    "needs_reply": False,
                })
        return json.dumps({"results": results})


def make_review(review_id: str, text: str, rating: int = 5) -> CleanedReview:
    return CleanedReview(
        review_id=review_id,
        product="Test Product",
        rating=rating,
        date="2026-10-01",
        review_text=text,
        text_hash=f"hash_{review_id}",
        injection_suspected=False,
    )


def test_rating_mismatch_logic():
    """Verify check_rating_mismatch rules."""
    # 5 stars but negative -> mismatch
    assert check_rating_mismatch(5, "negative") is True
    # 5 stars and positive -> match
    assert check_rating_mismatch(5, "positive") is False
    # 1 star and negative -> match
    assert check_rating_mismatch(1, "negative") is False
    # 1 star and positive -> mismatch
    assert check_rating_mismatch(1, "positive") is True
    # 3 stars and neutral -> match
    assert check_rating_mismatch(3, "neutral") is False
    # 3 stars and negative -> mismatch
    assert check_rating_mismatch(3, "negative") is True
    # Unknown rating (None) -> never mismatch
    assert check_rating_mismatch(None, "negative") is False
    assert check_rating_mismatch(None, "positive") is False


def test_analyzer_retries_invalid_response_once_then_marks_analysis_failed():
    """Verify analyzer retries an invalid output once; if still bad, marks analysis_failed."""
    # Item 1 returns invalid sentiment "superb" and missing topics
    bad_item = {
        "id": 1,
        "sentiment": "superb",  # Invalid enum
        "topics": [],           # Invalid empty list
        "summary": "Great",
        "needs_reply": False,
    }
    mock_llm = MockLLM(override_responses={1: bad_item})
    cache = Cache(":memory:")
    analyzer = ReviewAnalyzer(llm=mock_llm, cache=cache, batch_size=5)

    reviews = [make_review("R001", "Some product text")]
    result = analyzer.analyze(reviews)

    # Initial call + 1 retry = 2 calls
    assert mock_llm.call_count == 2
    assert len(result.reviews) == 1

    failed_rev = result.reviews[0]
    assert failed_rev.review_id == "R001"
    assert failed_rev.status == "analysis_failed"
    assert failed_rev.sentiment is None
    assert failed_rev.topics == []
    assert failed_rev.summary is None
    assert failed_rev.needs_reply is False

    # Failed item must NOT be cached
    assert cache.get(f"review:v1.0:{mock_llm.settings.model}:hash_R001") is None


def test_cache_second_run_makes_zero_calls():
    """Verify that re-analyzing already cached reviews makes zero external LLM calls."""
    mock_llm = MockLLM()
    cache = Cache(":memory:")
    analyzer = ReviewAnalyzer(llm=mock_llm, cache=cache, batch_size=5)

    reviews = [
        make_review("R001", "Text 1"),
        make_review("R002", "Text 2"),
        make_review("R003", "Text 3"),
    ]

    # Run 1: Uncached -> calls LLM
    res1 = analyzer.analyze(reviews)
    assert res1.ai_calls == 1
    assert res1.cache_hits == 0
    assert all(r.status == "ok" for r in res1.reviews)

    # Run 2: Same reviews -> 100% cache hits, zero AI calls
    mock_llm.call_count = 0
    res2 = analyzer.analyze(reviews)
    assert res2.ai_calls == 0
    assert res2.cache_hits == 3
    assert mock_llm.call_count == 0
    assert len(res2.reviews) == 3
