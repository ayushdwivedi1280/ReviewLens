"""Pydantic schemas and JSON parsing helpers for review analysis and insights.

Enforces strict validation, forbidden extra fields, and exact enumeration
of sentiments and topics to reject malformed or hijacked LLM responses.
"""

from typing import List, Literal, Any, Set
import json
import re
from pydantic import BaseModel, ConfigDict, Field, field_validator

SentimentType = Literal["positive", "neutral", "negative"]

TopicType = Literal[
    "quality",
    "delivery",
    "price",
    "packaging",
    "support",
    "size_fit",
    "other",
]

ALLOWED_SENTIMENTS: Set[str] = {"positive", "neutral", "negative"}
ALLOWED_TOPICS: Set[str] = {
    "quality",
    "delivery",
    "price",
    "packaging",
    "support",
    "size_fit",
    "other",
}


class StrictBaseModel(BaseModel):
    """Base model that forbids extra fields and enforces strict validation."""
    model_config = ConfigDict(extra="forbid", strict=True)


class ReviewAnalysis(StrictBaseModel):
    """Classification schema for an individual customer review."""
    sentiment: SentimentType
    topics: List[TopicType] = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=500)
    needs_reply: bool

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Summary must not be empty.")
        # Ensure single-line summary
        return " ".join(s.split())

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, topics: List[TopicType]) -> List[TopicType]:
        if not topics:
            raise ValueError("At least one topic is required.")
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for t in topics:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return deduped


class BatchAnalysisItem(ReviewAnalysis):
    """Batch item returned by LLM with an integer batch-local ID."""
    id: int


class BatchAnalysisResponse(StrictBaseModel):
    """Full batch response payload from LLM."""
    results: List[BatchAnalysisItem]


class TopInsightItem(StrictBaseModel):
    """Individual theme card in aggregate insights."""
    title: str = Field(min_length=1)
    count: int = Field(ge=0)
    review_ids: List[str]


class TopInsights(StrictBaseModel):
    """Aggregate top 3 complaints and top 3 praises across products."""
    top_complaints: List[TopInsightItem] = Field(default_factory=list, max_length=3)
    top_praises: List[TopInsightItem] = Field(default_factory=list, max_length=3)


def load_json(raw_text: str) -> Any:
    """Extract and parse JSON from raw LLM output.

    Strips markdown code fences (```json ... ```) and isolates JSON objects.
    Raises ValueError on syntax failure.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty response cannot be parsed as JSON.")

    text = raw_text.strip()

    # Strip markdown code fences if present
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
        else:
            # Fallback fence stripping
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: locate outermost brackets
        start_obj = text.find("{")
        end_obj = text.rfind("}")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            try:
                return json.loads(text[start_obj:end_obj + 1])
            except json.JSONDecodeError:
                pass
        start_arr = text.find("[")
        end_arr = text.rfind("]")
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            try:
                return json.loads(text[start_arr:end_arr + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse valid JSON from LLM response: {raw_text[:100]}...")
