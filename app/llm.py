"""OpenAI-compatible LLM client with exponential backoff and jitter.

Supports Groq and Gemini OpenAI-compatible endpoints with injectable sleep for instant testing.
Handles 429 rate limits, 5xx transient server errors, and transport timeouts.
"""

from typing import Optional, Callable, Dict, Any
import random
import time
import httpx
from app.config import Settings


class LLMUnavailable(Exception):
    """Raised when the LLM service cannot be reached after exhausting retries."""
    pass


class LLMRequestRejected(Exception):
    """Raised when the LLM request is permanently rejected (401, 403, invalid model)."""
    pass


def parse_retry_after(header_val: Optional[str]) -> Optional[float]:
    """Parse Retry-After header as seconds float if possible."""
    if not header_val:
        return None
    try:
        val = float(header_val.strip())
        return max(0.0, val)
    except (ValueError, TypeError):
        return None


def backoff_delay(attempt: int, retry_after: Optional[float] = None) -> float:
    """Calculate exponential backoff delay with jitter.

    attempt: 1-indexed attempt number (1, 2, 3...)
    Delays: 1s, 2s, 4s, 8s, 16s... capped at 30s + up to 25% jitter.
    Honours retry_after if larger than calculated delay.
    """
    base = min(30.0, float(2 ** (attempt - 1)))
    jitter = base * random.uniform(0.0, 0.25)
    total = base + jitter
    if retry_after is not None and retry_after > total:
        return retry_after
    return total


class LLMClient:
    """HTTP client communicating with OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        settings: Settings,
        client: Optional[httpx.Client] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.settings = settings
        self._external_client = client is not None
        self.client = client or httpx.Client(timeout=60.0)
        self.sleep_fn = sleep_fn or time.sleep

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        json_mode: bool = True,
    ) -> str:
        """Send completion request to LLM with retries on transient errors."""
        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": self.settings.temperature,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error_msg = ""

        for attempt in range(1, self.settings.max_retries + 1):
            try:
                response = self.client.post(url, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error_msg = f"Network/timeout error: {exc}"
                if attempt == self.settings.max_retries:
                    break
                delay = backoff_delay(attempt)
                self.sleep_fn(delay)
                continue

            # Success
            if response.status_code == 200:
                try:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, ValueError) as exc:
                    raise LLMUnavailable(f"Malformed response payload from LLM API: {exc}")

            # Transient retryable status codes: 408, 429, 500, 502, 503, 504
            if response.status_code in {408, 429, 500, 502, 503, 504}:
                last_error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                if attempt == self.settings.max_retries:
                    break
                retry_after_hdr = parse_retry_after(response.headers.get("Retry-After"))
                delay = backoff_delay(attempt, retry_after=retry_after_hdr)
                self.sleep_fn(delay)
                continue

            # Non-retryable client errors (400, 401, 403, 404, etc.)
            raise LLMRequestRejected(
                f"LLM API request permanently rejected with HTTP {response.status_code}: {response.text[:300]}"
            )

        raise LLMUnavailable(
            f"LLM service unavailable after {self.settings.max_retries} attempts. Last error: {last_error_msg}"
        )

    def close(self) -> None:
        """Close underlying client if owned."""
        if not self._external_client:
            self.client.close()
