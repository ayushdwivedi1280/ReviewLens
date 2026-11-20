"""Aggregate store insights: Top 3 complaints and top 3 praises.

Queries the LLM with all classified reviews, validates against TopInsights schema,
drops invalid IDs, retries once on failure, and falls back to a deterministic topic-frequency
summary if LLM calls fail. Cached by hash of all inputs for instant replay.
"""

from typing import List, Dict, Any, Optional
from collections import Counter
import hashlib
import json

from app.schema import (
    TopInsights,
    TopInsightItem,
    load_json,
)
from app.prompts import (
    PROMPT_VERSION,
    INSIGHTS_SYSTEM_PROMPT,
    build_insights_user_message,
)
from app.cache import Cache
from app.llm import LLMClient, LLMUnavailable, LLMRequestRejected


def compute_insights_hash(reviews_data: List[Dict[str, Any]]) -> str:
    """Compute hash of review IDs and summaries for cache keying."""
    keys = [
        f"{r.get('review_id')}:{r.get('sentiment')}:{r.get('summary')}"
        for r in sorted(reviews_data, key=lambda x: str(x.get("review_id")))
    ]
    raw = "|".join(keys)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_deterministic_insights(reviews_data: List[Dict[str, Any]]) -> TopInsights:
    """Deterministic fallback: aggregate top topics for positive and negative reviews."""
    valid_ids = {r["review_id"] for r in reviews_data if "review_id" in r}

    neg_topics: Counter = Counter()
    neg_map: Dict[str, List[str]] = {}
    pos_topics: Counter = Counter()
    pos_map: Dict[str, List[str]] = {}

    for r in reviews_data:
        rid = r.get("review_id")
        sentiment = r.get("sentiment")
        topics = r.get("topics", [])
        if not rid or not sentiment:
            continue

        if sentiment == "negative":
            for t in topics:
                neg_topics[t] += 1
                neg_map.setdefault(t, []).append(rid)
        elif sentiment == "positive":
            for t in topics:
                pos_topics[t] += 1
                pos_map.setdefault(t, []).append(rid)

    top_complaints: List[TopInsightItem] = []
    for topic, count in neg_topics.most_common(3):
        label = topic.replace("_", " ").title()
        top_complaints.append(
            TopInsightItem(
                title=f"Issues with {label}",
                count=count,
                review_ids=[rid for rid in neg_map.get(topic, []) if rid in valid_ids],
            )
        )

    top_praises: List[TopInsightItem] = []
    for topic, count in pos_topics.most_common(3):
        label = topic.replace("_", " ").title()
        top_praises.append(
            TopInsightItem(
                title=f"Satisfaction with {label}",
                count=count,
                review_ids=[rid for rid in pos_map.get(topic, []) if rid in valid_ids],
            )
        )

    return TopInsights(
        top_complaints=top_complaints,
        top_praises=top_praises,
    )


def extract_top_insights(
    llm: LLMClient,
    cache: Cache,
    analyzed_reviews: List[Dict[str, Any]],
) -> tuple[TopInsights, int]:
    """Generate top 3 complaints and praises via LLM or deterministic fallback.

    Returns (TopInsights, ai_calls_made).
    """
    # Filter for successfully analyzed reviews
    ok_reviews = [
        r for r in analyzed_reviews if r.get("status") == "ok" and r.get("sentiment")
    ]
    if not ok_reviews:
        return TopInsights(), 0

    valid_ids = {r["review_id"] for r in ok_reviews}
    model_name = llm.settings.model
    ihash = compute_insights_hash(ok_reviews)
    ckey = f"insights:{PROMPT_VERSION}:{model_name}:{ihash}"

    # Check cache
    cached_val = cache.get(ckey)
    if cached_val:
        try:
            return TopInsights.model_validate_json(cached_val), 0
        except Exception:
            pass

    user_msg = build_insights_user_message(ok_reviews)
    ai_calls = 0

    def parse_insights(raw: str) -> Optional[TopInsights]:
        try:
            parsed = load_json(raw)
            if not isinstance(parsed, dict):
                return None

            complaints = parsed.get("top_complaints", [])
            praises = parsed.get("top_praises", [])

            # Filter valid review_ids
            cleaned_complaints = []
            for item in complaints[:3]:
                if not isinstance(item, dict):
                    continue
                rids = [rid for rid in item.get("review_ids", []) if rid in valid_ids]
                cleaned_complaints.append(
                    TopInsightItem(
                        title=str(item.get("title", "Customer Complaint")),
                        count=len(rids) if rids else int(item.get("count", 0)),
                        review_ids=rids,
                    )
                )

            cleaned_praises = []
            for item in praises[:3]:
                if not isinstance(item, dict):
                    continue
                rids = [rid for rid in item.get("review_ids", []) if rid in valid_ids]
                cleaned_praises.append(
                    TopInsightItem(
                        title=str(item.get("title", "Customer Praise")),
                        count=len(rids) if rids else int(item.get("count", 0)),
                        review_ids=rids,
                    )
                )

            return TopInsights(
                top_complaints=cleaned_complaints,
                top_praises=cleaned_praises,
            )
        except Exception:
            return None

    # First attempt
    try:
        raw_res = llm.complete(
            system_prompt=INSIGHTS_SYSTEM_PROMPT,
            user_message=user_msg,
            json_mode=True,
        )
        ai_calls += 1
        insights = parse_insights(raw_res)
        if insights:
            cache.set(ckey, insights.model_dump_json())
            return insights, ai_calls
    except (LLMUnavailable, Exception):
        ai_calls += 1

    # Single retry attempt
    try:
        retry_res = llm.complete(
            system_prompt=INSIGHTS_SYSTEM_PROMPT,
            user_message=user_msg,
            json_mode=True,
        )
        ai_calls += 1
        insights = parse_insights(retry_res)
        if insights:
            cache.set(ckey, insights.model_dump_json())
            return insights, ai_calls
    except (LLMUnavailable, Exception):
        pass

    # Deterministic fallback on failures
    fallback = generate_deterministic_insights(ok_reviews)
    cache.set(ckey, fallback.model_dump_json())
    return fallback, ai_calls
