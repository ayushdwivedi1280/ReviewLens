"""Batched review analyzer with validation, single retry, and fallback to analysis_failed.

Processes reviews sequentially in batches of BATCH_SIZE (default 5).
Checks SQLite cache before making external LLM calls.
Never crashes on bad AI outputs; marks failed rows as 'analysis_failed'.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import json

from app.cleaning import CleanedReview
from app.schema import (
    ReviewAnalysis,
    BatchAnalysisResponse,
    load_json,
)
from app.prompts import (
    PROMPT_VERSION,
    ANALYSIS_SYSTEM_PROMPT,
    build_analysis_user_message,
)
from app.cache import Cache
from app.llm import LLMClient, LLMUnavailable, LLMRequestRejected


@dataclass
class AnalyzedReview:
    """Classified review record with metadata, flags, and mismatch detection."""
    review_id: str
    product: str
    rating: Optional[int]
    date: str
    review_text: str
    status: str  # "ok" or "analysis_failed"
    sentiment: Optional[str]
    topics: List[str]
    summary: Optional[str]
    needs_reply: bool
    rating_mismatch: bool
    injection_suspected: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalyzerResult:
    """Output of batch analysis across all unique reviews."""
    reviews: List[AnalyzedReview]
    ai_calls: int
    cache_hits: int


def check_rating_mismatch(rating: Optional[int], sentiment: Optional[str]) -> bool:
    """Check if star rating conflicts with AI classified sentiment.

    Rules:
    - rating >= 4 expects positive (e.g. 5-star negative is a mismatch)
    - rating == 3 expects neutral
    - rating <= 2 expects negative
    - missing/unknown rating is never a mismatch
    """
    if rating is None or sentiment is None:
        return False
    if rating >= 4 and sentiment != "positive":
        return True
    if rating == 3 and sentiment != "neutral":
        return True
    if rating <= 2 and sentiment != "negative":
        return True
    return False


def make_cache_key(model: str, text_hash: str) -> str:
    """Construct deterministic cache key including prompt version and model name."""
    return f"review:{PROMPT_VERSION}:{model}:{text_hash}"


class ReviewAnalyzer:
    """Coordinates batching, caching, LLM execution, schema validation, and fallback."""

    def __init__(
        self,
        llm: LLMClient,
        cache: Cache,
        batch_size: int = 5,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.batch_size = max(1, batch_size)

    def analyze(self, reviews: List[CleanedReview]) -> AnalyzerResult:
        """Analyze a list of cleaned unique reviews.

        Returns structured AnalyzedReview records, tracking AI calls and cache hits.
        """
        ai_calls = 0
        cache_hits = 0
        model_name = self.llm.settings.model

        analyzed_map: Dict[str, AnalyzedReview] = {}
        uncached: List[CleanedReview] = []

        # 1. Check Cache first
        for rev in reviews:
            ckey = make_cache_key(model_name, rev.text_hash)
            cached_val = self.cache.get(ckey)
            if cached_val:
                try:
                    data = json.loads(cached_val)
                    analysis = ReviewAnalysis.model_validate(data)
                    mismatch = check_rating_mismatch(rev.rating, analysis.sentiment)
                    analyzed_map[rev.review_id] = AnalyzedReview(
                        review_id=rev.review_id,
                        product=rev.product,
                        rating=rev.rating,
                        date=rev.date,
                        review_text=rev.review_text,
                        status="ok",
                        sentiment=analysis.sentiment,
                        topics=analysis.topics,
                        summary=analysis.summary,
                        needs_reply=analysis.needs_reply,
                        rating_mismatch=mismatch,
                        injection_suspected=rev.injection_suspected,
                    )
                    cache_hits += 1
                    continue
                except Exception:
                    # Stale or corrupted cache entry, re-analyze
                    pass

            uncached.append(rev)

        # 2. Process uncached reviews in sequential batches
        for i in range(0, len(uncached), self.batch_size):
            chunk = uncached[i : i + self.batch_size]
            chunk_results, calls = self._process_batch(chunk, model_name)
            ai_calls += calls
            for item in chunk_results:
                analyzed_map[item.review_id] = item

        # Maintain original review ordering
        ordered_reviews = [analyzed_map[r.review_id] for r in reviews if r.review_id in analyzed_map]

        return AnalyzerResult(
            reviews=ordered_reviews,
            ai_calls=ai_calls,
            cache_hits=cache_hits,
        )

    def _process_batch(
        self,
        chunk: List[CleanedReview],
        model_name: str,
    ) -> tuple[List[AnalyzedReview], int]:
        """Process a single batch: call LLM, validate, retry failures once."""
        ai_calls = 0
        id_to_rev: Dict[int, CleanedReview] = {
            idx: rev for idx, rev in enumerate(chunk, start=1)
        }
        batch_items = [
            {"id": idx, "product": rev.product, "text": rev.review_text}
            for idx, rev in id_to_rev.items()
        ]

        user_message = build_analysis_user_message(batch_items)
        succeeded: Dict[int, ReviewAnalysis] = {}
        failed_ids: List[int] = []

        # First attempt
        try:
            raw_response = self.llm.complete(
                system_prompt=ANALYSIS_SYSTEM_PROMPT,
                user_message=user_message,
                json_mode=True,
            )
            ai_calls += 1
            succeeded, failed_ids = self._parse_and_validate_results(raw_response, id_to_rev)
        except LLMUnavailable:
            ai_calls += 1
            failed_ids = list(id_to_rev.keys())
        except LLMRequestRejected:
            # Fatal client error (401, 403, bad model) -> re-raise
            raise

        # Retry once for any missing or invalid items
        if failed_ids:
            retry_items = [
                {"id": local_id, "product": id_to_rev[orig_id].product, "text": id_to_rev[orig_id].review_text}
                for local_id, orig_id in enumerate(failed_ids, start=1)
            ]
            local_to_orig = {
                local_id: orig_id for local_id, orig_id in enumerate(failed_ids, start=1)
            }
            retry_user_msg = build_analysis_user_message(retry_items)

            try:
                retry_response = self.llm.complete(
                    system_prompt=ANALYSIS_SYSTEM_PROMPT,
                    user_message=retry_user_msg,
                    json_mode=True,
                )
                ai_calls += 1
                retry_id_to_rev = {
                    local_id: id_to_rev[orig_id] for local_id, orig_id in local_to_orig.items()
                }
                retry_succeeded, _ = self._parse_and_validate_results(
                    retry_response, retry_id_to_rev
                )
                for local_id, analysis in retry_succeeded.items():
                    orig_id = local_to_orig[local_id]
                    succeeded[orig_id] = analysis
            except (LLMUnavailable, Exception):
                # Retry failed; items will fall back to analysis_failed
                pass

        # Assemble final results for chunk
        results: List[AnalyzedReview] = []
        for idx, rev in id_to_rev.items():
            if idx in succeeded:
                analysis = succeeded[idx]
                mismatch = check_rating_mismatch(rev.rating, analysis.sentiment)
                # Store in cache
                ckey = make_cache_key(model_name, rev.text_hash)
                self.cache.set(ckey, analysis.model_dump_json())

                results.append(
                    AnalyzedReview(
                        review_id=rev.review_id,
                        product=rev.product,
                        rating=rev.rating,
                        date=rev.date,
                        review_text=rev.review_text,
                        status="ok",
                        sentiment=analysis.sentiment,
                        topics=analysis.topics,
                        summary=analysis.summary,
                        needs_reply=analysis.needs_reply,
                        rating_mismatch=mismatch,
                        injection_suspected=rev.injection_suspected,
                    )
                )
            else:
                # Mark as analysis_failed
                results.append(
                    AnalyzedReview(
                        review_id=rev.review_id,
                        product=rev.product,
                        rating=rev.rating,
                        date=rev.date,
                        review_text=rev.review_text,
                        status="analysis_failed",
                        sentiment=None,
                        topics=[],
                        summary=None,
                        needs_reply=False,
                        rating_mismatch=False,
                        injection_suspected=rev.injection_suspected,
                    )
                )

        return results, ai_calls

    def _parse_and_validate_results(
        self,
        raw_response: str,
        id_to_rev: Dict[int, CleanedReview],
    ) -> tuple[Dict[int, ReviewAnalysis], List[int]]:
        """Parse JSON response and validate each item against ReviewAnalysis schema."""
        succeeded: Dict[int, ReviewAnalysis] = {}
        failed_ids: List[int] = []

        try:
            parsed = load_json(raw_response)
            if not isinstance(parsed, dict) or "results" not in parsed:
                return {}, list(id_to_rev.keys())

            items = parsed["results"]
            if not isinstance(items, list):
                return {}, list(id_to_rev.keys())

            for item in items:
                if not isinstance(item, dict) or "id" not in item:
                    continue
                item_id = item["id"]
                if item_id not in id_to_rev:
                    continue

                try:
                    # Validate individual item
                    analysis = ReviewAnalysis(
                        sentiment=item.get("sentiment"),
                        topics=item.get("topics"),
                        summary=item.get("summary"),
                        needs_reply=item.get("needs_reply"),
                    )
                    succeeded[item_id] = analysis
                except Exception:
                    # Schema validation failed for this item
                    pass

        except Exception:
            return {}, list(id_to_rev.keys())

        # Determine which expected IDs failed or were missing
        for expected_id in id_to_rev:
            if expected_id not in succeeded:
                failed_ids.append(expected_id)

        return succeeded, failed_ids
