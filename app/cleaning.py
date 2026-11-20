"""CSV cleaning and normalization for customer product reviews.

Handles bad rows gracefully: HTML stripping, whitespace normalization,
duplicate detection via casefolded text hashing, rating validation (1-5),
and heuristic prompt injection flagging for display.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
import hashlib
import html
import io
import csv
import re

REQUIRED_COLUMNS = ["review_id", "product", "rating", "date", "review_text"]

INJECTION_PATTERN = re.compile(
    r"(?i)(ignore\s+(all\s+)?previous\s+instructions|system\s+prompt|"
    r"disregard\s+(all\s+)?previous|mark\s+this\s+review\s+as)"
)


class CleaningError(Exception):
    """Raised when CSV structure or encoding is invalid."""
    pass


@dataclass
class CleanedReview:
    """A valid, non-duplicate review ready for LLM analysis."""
    review_id: str
    product: str
    rating: Optional[int]  # 1-5 or None for missing/invalid rating
    date: str
    review_text: str
    text_hash: str  # SHA-256 of casefolded cleaned text
    injection_suspected: bool = False
    duplicate_of: Optional[str] = None


@dataclass
class SkippedReview:
    """A review dropped prior to analysis (e.g. empty text)."""
    review_id: str
    product: str
    rating: Optional[int]
    date: str
    original_text: str
    reason: str


@dataclass
class CleanedData:
    """Result of parsing and cleaning an uploaded CSV."""
    reviews: List[CleanedReview]
    skipped: List[SkippedReview]
    duplicates: List[CleanedReview]


def clean_text(raw_text: str) -> str:
    """Sanitize review text: strip HTML, unescape entities, collapse whitespace.

    Keeps Unicode, Hindi/Devanagari, and emoji intact. Capped at 2000 chars.
    """
    if not raw_text:
        return ""
    # Unescape HTML entities (e.g. &amp; -> &)
    text = html.unescape(raw_text)
    # Strip HTML tags like <b> or <p>
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2000]


def parse_rating(val: Optional[str]) -> Optional[int]:
    """Parse 1-5 integer rating. Returns None for missing or invalid ratings."""
    if val is None:
        return None
    val_clean = str(val).strip()
    if not val_clean:
        return None
    try:
        r = int(val_clean)
        if 1 <= r <= 5:
            return r
    except (ValueError, TypeError):
        pass
    return None


def compute_text_hash(text: str) -> str:
    """Return SHA-256 hex digest of casefolded cleaned text for deduplication."""
    return hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()


def check_injection_suspected(text: str) -> bool:
    """Check text against prompt injection heuristic regex for display alert."""
    return bool(INJECTION_PATTERN.search(text))


def parse_reviews(file_bytes: bytes) -> CleanedData:
    """Parse and clean reviews CSV bytes.

    Never crashes on bad rows.
    Raises CleaningError if required columns are missing.
    """
    try:
        decoded = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CleaningError(f"File encoding error: {exc}. Must be valid UTF-8.") from exc

    f = io.StringIO(decoded)
    reader = csv.DictReader(f)

    if reader.fieldnames is None:
        raise CleaningError("Empty CSV file or missing header row.")

    # Clean header column names
    fieldnames = [col.strip().lower() for col in reader.fieldnames if col]
    missing = [col for col in REQUIRED_COLUMNS if col not in fieldnames]
    if missing:
        raise CleaningError(f"Missing required CSV columns: {', '.join(missing)}")

    # Map normalized header back to DictReader fields
    header_map = {col.strip().lower(): col for col in reader.fieldnames if col}

    reviews: List[CleanedReview] = []
    skipped: List[SkippedReview] = []
    duplicates: List[CleanedReview] = []

    seen_hashes: Dict[str, str] = {}  # text_hash -> review_id
    seen_ids: Dict[str, int] = {}     # review_id -> occurrence count

    for row_idx, row in enumerate(reader, start=1):
        raw_id = (row.get(header_map["review_id"]) or f"ROW_{row_idx}").strip()
        product = (row.get(header_map["product"]) or "Unknown Product").strip()
        raw_rating = row.get(header_map["rating"])
        date = (row.get(header_map["date"]) or "").strip()
        raw_text = row.get(header_map["review_text"]) or ""

        # Normalize ID to prevent collisions
        if raw_id in seen_ids:
            seen_ids[raw_id] += 1
            review_id = f"{raw_id}_dup{seen_ids[raw_id]}"
        else:
            seen_ids[raw_id] = 1
            review_id = raw_id

        rating = parse_rating(raw_rating)
        cleaned_text = clean_text(raw_text)

        # Empty text check
        if not cleaned_text:
            skipped.append(SkippedReview(
                review_id=review_id,
                product=product,
                rating=rating,
                date=date,
                original_text=raw_text,
                reason="empty_text"
            ))
            continue

        thash = compute_text_hash(cleaned_text)
        injection_flag = check_injection_suspected(cleaned_text)

        # Duplicate check (same cleaned text, case-insensitive)
        if thash in seen_hashes:
            orig_id = seen_hashes[thash]
            duplicates.append(CleanedReview(
                review_id=review_id,
                product=product,
                rating=rating,
                date=date,
                review_text=cleaned_text,
                text_hash=thash,
                injection_suspected=injection_flag,
                duplicate_of=orig_id
            ))
            continue

        seen_hashes[thash] = review_id
        reviews.append(CleanedReview(
            review_id=review_id,
            product=product,
            rating=rating,
            date=date,
            review_text=cleaned_text,
            text_hash=thash,
            injection_suspected=injection_flag,
            duplicate_of=None
        ))

    return CleanedData(reviews=reviews, skipped=skipped, duplicates=duplicates)
