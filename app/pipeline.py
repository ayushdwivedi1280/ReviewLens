"""End-to-end review analysis pipeline with timing, metrics, and in-memory run store."""

from dataclasses import asdict
from typing import Dict, Any, Optional
import time
import uuid

from app.cleaning import parse_reviews, CleaningError
from app.analyzer import ReviewAnalyzer
from app.insights import extract_top_insights
from app.cache import Cache
from app.llm import LLMClient

# In-memory store mapping run_id -> {review_id -> dict of review details}
RUN_STORE: Dict[str, Dict[str, Dict[str, Any]]] = {}


def analyze_file(
    file_bytes: bytes,
    llm: LLMClient,
    cache: Cache,
    batch_size: int = 5,
) -> Dict[str, Any]:
    """Execute full analysis pipeline on uploaded CSV file bytes.

    Steps:
    1. Parse and sanitize CSV (strip HTML, deduplicate, validate ratings).
    2. Batch analyze uncached reviews via LLM with single-retry fallback.
    3. Synthesize top 3 complaints and top 3 praises.
    4. Compute summary aggregations, rating-sentiment agreement %, and performance metrics.
    5. Cache run details in RUN_STORE for on-demand suggested replies.
    """
    start_time = time.perf_counter()

    # 1. Clean data
    cleaned = parse_reviews(file_bytes)

    # 2. Analyze reviews
    analyzer = ReviewAnalyzer(llm=llm, cache=cache, batch_size=batch_size)
    analyzer_res = analyzer.analyze(cleaned.reviews)

    # 3. Extract top insights
    reviews_dicts = [r.to_dict() for r in analyzer_res.reviews]
    insights, insight_calls = extract_top_insights(llm, cache, reviews_dicts)

    # 4. Aggregations & Metrics
    total_analyzed = len(analyzer_res.reviews)
    ok_reviews = [r for r in analyzer_res.reviews if r.status == "ok"]

    pos_count = sum(1 for r in ok_reviews if r.sentiment == "positive")
    neu_count = sum(1 for r in ok_reviews if r.sentiment == "neutral")
    neg_count = sum(1 for r in ok_reviews if r.sentiment == "negative")
    failed_count = sum(1 for r in analyzer_res.reviews if r.status == "analysis_failed")
    needs_reply_count = sum(1 for r in analyzer_res.reviews if r.needs_reply)

    if total_analyzed > 0:
        pos_pct = round((pos_count / total_analyzed) * 100, 1)
        neu_pct = round((neu_count / total_analyzed) * 100, 1)
        neg_pct = round((neg_count / total_analyzed) * 100, 1)
    else:
        pos_pct = 0.0
        neu_pct = 0.0
        neg_pct = 0.0

    # Rating Mismatch Agreement %
    known_ratings = [r for r in ok_reviews if r.rating is not None]
    if known_ratings:
        agreed_count = sum(1 for r in known_ratings if not r.rating_mismatch)
        rating_agreement_pct = round((agreed_count / len(known_ratings)) * 100, 1)
        mismatch_count = len(known_ratings) - agreed_count
    else:
        rating_agreement_pct = 100.0
        mismatch_count = 0

    elapsed_seconds = round(time.perf_counter() - start_time, 2)
    total_ai_calls = analyzer_res.ai_calls + insight_calls

    run_id = uuid.uuid4().hex[:12]

    # Save to run store for on-click reply lookup
    RUN_STORE[run_id] = {
        r.review_id: {
            "product": r.product,
            "text": r.review_text,
            "sentiment": r.sentiment or "neutral",
            "summary": r.summary or "",
        }
        for r in analyzer_res.reviews
    }

    return {
        "run_id": run_id,
        "summary": {
            "total_analyzed": total_analyzed,
            "positive_count": pos_count,
            "positive_pct": pos_pct,
            "neutral_count": neu_count,
            "neutral_pct": neu_pct,
            "negative_count": neg_count,
            "negative_pct": neg_pct,
            "needs_reply_count": needs_reply_count,
            "failed_count": failed_count,
            "rating_agreement_pct": rating_agreement_pct,
            "rating_mismatches_count": mismatch_count,
        },
        "insights": insights.model_dump(),
        "reviews": reviews_dicts,
        "skipped": [asdict(s) for s in cleaned.skipped],
        "duplicates": [asdict(d) for d in cleaned.duplicates],
        "metrics": {
            "ai_calls": total_ai_calls,
            "cache_hits": analyzer_res.cache_hits,
            "seconds": elapsed_seconds,
        },
    }
