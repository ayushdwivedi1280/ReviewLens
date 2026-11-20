"""End-to-end integration tests for FastAPI REST endpoints using TestClient and FakeLLM."""

from pathlib import Path
import json
import re
from typing import Dict, Any
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.config import Settings
from app.llm import LLMClient
from app.cache import Cache

SAMPLE_CSV_PATH = Path(__file__).resolve().parent.parent / "reviews.csv"


class FakeLLM(LLMClient):
    """Deterministic FakeLLM that produces realistic structured responses."""

    def __init__(self) -> None:
        self.settings = Settings(
            provider="groq",
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
            api_key="test-api-key",
            batch_size=5,
        )
        self.call_count = 0

    def complete(self, system_prompt: str, user_message: str, json_mode: bool = True) -> str:
        self.call_count += 1

        # Suggested Reply handler
        if "store owner replying directly" in system_prompt:
            return (
                "Thank you for your feedback. We apologize for the trouble with your order "
                "and would appreciate the chance to resolve this. Please contact our support "
                "team with your order number so we can help."
            )

        # Top Insights handler
        if "ReviewLens Insights Engine" in system_prompt:
            return json.dumps({
                "top_complaints": [
                    {"title": "Broken handles and fragile build", "count": 2, "review_ids": ["R008"]},
                    {"title": "Delayed delivery times", "count": 2, "review_ids": ["R003", "R007"]},
                ],
                "top_praises": [
                    {"title": "Soft fabric and neat stitching", "count": 2, "review_ids": ["R001", "R028"]},
                    {"title": "High quality appearance", "count": 2, "review_ids": ["R006", "R024"]},
                ]
            })

        # Batch review classification handler
        match = re.search(r"<reviews>\s*(\[[\s\S]*?\])\s*</reviews>", user_message)
        items = json.loads(match.group(1)) if match else []

        results = []
        for it in items:
            text = it.get("text", "")
            lower = text.lower()

            # R012 & R015: prompt injection attempts -> must still be classified as negative
            if "ignore all previous instructions" in lower or "stopped working" in lower:
                results.append({
                    "id": it["id"],
                    "sentiment": "negative",
                    "topics": ["quality"],
                    "summary": "Left earbud stopped functioning after two days.",
                    "needs_reply": True,
                })
            # Sarcasm (R003) or severe issues
            elif "delivered in just 3 weeks" in lower or "terrible" in lower or "broken" in lower or "mud on the sole" in lower:
                results.append({
                    "id": it["id"],
                    "sentiment": "negative",
                    "topics": ["delivery" if "3 weeks" in lower else "quality"],
                    "summary": "Customer experienced severe dissatisfaction with product or delivery.",
                    "needs_reply": True,
                })
            # Positive reviews
            elif "super soft" in lower or "beautiful" in lower or "love it" in lower or "best shoes" in lower:
                results.append({
                    "id": it["id"],
                    "sentiment": "positive",
                    "topics": ["quality"],
                    "summary": "Customer is extremely satisfied with product quality.",
                    "needs_reply": False,
                })
            # Neutral / general
            else:
                results.append({
                    "id": it["id"],
                    "sentiment": "neutral",
                    "topics": ["price" if "overpriced" in lower else "quality"],
                    "summary": "Customer provided moderate feedback.",
                    "needs_reply": False,
                })

        return json.dumps({"results": results})


@pytest.fixture
def test_client():
    fake_llm = FakeLLM()
    cache = Cache(":memory:")
    app = create_app(llm=fake_llm, cache=cache)
    return TestClient(app)


def test_health_check(test_client):
    """Verify /api/health endpoint returns 200 OK."""
    resp = test_client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "app": "ReviewLens"}


def test_serve_dashboard_index(test_client):
    """Verify root / serves static index.html."""
    resp = test_client.get("/")
    assert resp.status_code == 200
    assert "ReviewLens" in resp.text
    assert "<!DOCTYPE html>" in resp.text


def test_analyze_sample_csv_end_to_end_and_cache(test_client):
    """Verify full upload pipeline, correct review counts, and 0 AI calls on re-upload."""
    csv_bytes = SAMPLE_CSV_PATH.read_bytes()

    # 1. First Upload: processes uncached reviews
    files = {"file": ("reviews.csv", csv_bytes, "text/csv")}
    resp = test_client.post("/api/analyze", files=files)
    assert resp.status_code == 200
    data = resp.json()

    # Validate output structure
    assert "run_id" in data
    assert "summary" in data
    assert "insights" in data
    assert "reviews" in data
    assert "skipped" in data
    assert "duplicates" in data
    assert "metrics" in data

    # Verify counts
    assert data["summary"]["total_analyzed"] == 28
    assert len(data["reviews"]) == 28
    assert len(data["skipped"]) == 1
    assert data["skipped"][0]["review_id"] == "R005"
    assert len(data["duplicates"]) == 1
    assert data["duplicates"][0]["review_id"] == "R015"
    assert data["duplicates"][0]["duplicate_of"] == "R012"

    # Verify R012 injection resisted: negative, needs_reply true
    r012 = next(r for r in data["reviews"] if r["review_id"] == "R012")
    assert r012["sentiment"] == "negative"
    assert r012["needs_reply"] is True
    assert "quality" in r012["topics"]
    assert r012["injection_suspected"] is True

    # Verify R008 rating mismatch (5 stars vs negative sentiment)
    r008 = next(r for r in data["reviews"] if r["review_id"] == "R008")
    assert r008["rating"] == 5
    assert r008["sentiment"] == "negative"
    assert r008["rating_mismatch"] is True

    # Verify metrics on initial run
    assert data["metrics"]["ai_calls"] > 0
    assert data["metrics"]["cache_hits"] == 0

    # 2. Second Upload: Same CSV must produce 0 AI calls (100% cache hits)
    files2 = {"file": ("reviews.csv", csv_bytes, "text/csv")}
    resp2 = test_client.post("/api/analyze", files=files2)
    assert resp2.status_code == 200
    data2 = resp2.json()

    assert data2["metrics"]["ai_calls"] == 0
    assert data2["metrics"]["cache_hits"] == 28
    assert data2["summary"]["total_analyzed"] == 28

    # 3. Test on-demand reply generation for needs_reply row (R012)
    run_id = data["run_id"]
    reply_resp = test_client.post(
        "/api/reply",
        json={"run_id": run_id, "review_id": "R012"},
    )
    assert reply_resp.status_code == 200
    reply_data = reply_resp.json()
    assert "reply" in reply_data
    assert len(reply_data["reply"]) > 20
    assert "support" in reply_data["reply"].lower() or "order" in reply_data["reply"].lower()


def test_analyze_invalid_csv_returns_400(test_client):
    """Verify missing required columns returns HTTP 400."""
    bad_csv = b"review_id,rating\nR001,5\n"
    files = {"file": ("bad.csv", bad_csv, "text/csv")}
    resp = test_client.post("/api/analyze", files=files)
    assert resp.status_code == 400
    assert "Missing required CSV columns" in resp.json()["detail"]


def test_reply_missing_run_or_review_returns_404(test_client):
    """Verify non-existent run or review returns HTTP 404."""
    resp = test_client.post(
        "/api/reply",
        json={"run_id": "non_existent_run", "review_id": "R999"},
    )
    assert resp.status_code == 404
