"""On-demand merchant suggested reply generator.

Generated on user click for reviews flagged as 'needs_reply'.
Produces empathetic, concise, and professional responses under 70 words.
Does not invent fake coupon codes or monetary promises.
"""

from typing import Optional
from app.llm import LLMClient, LLMUnavailable, LLMRequestRejected
from app.prompts import (
    REPLY_SYSTEM_PROMPT,
    build_reply_user_message,
)


def generate_suggested_reply(
    llm: LLMClient,
    product: str,
    review_text: str,
    sentiment: str,
    summary: str,
) -> str:
    """Generate a store owner response to a customer review requiring assistance.

    Falls back to a polite deterministic template if LLM is unreachable.
    """
    user_msg = build_reply_user_message(
        product=product,
        text=review_text,
        sentiment=sentiment,
        summary=summary,
    )

    try:
        raw = llm.complete(
            system_prompt=REPLY_SYSTEM_PROMPT,
            user_message=user_msg,
            json_mode=False,
        )
        reply = raw.strip().strip('"').strip("'")
        if reply:
            return reply
    except Exception:
        pass

    # Deterministic fallback response adhering to constraints
    return (
        f"Thank you for sharing your experience with the {product}. "
        "We are truly sorry that this purchase fell short of your expectations. "
        "We want to resolve this for you right away—please reach out to our support "
        "team with your order number and details so we can make this right."
    )
