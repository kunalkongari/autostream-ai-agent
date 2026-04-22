"""
intent.py — Classifies incoming user messages into one of three intents.

Strategy: Two-pass approach
  1. Rule-based fast path (regex/keyword) — deterministic, zero latency.
  2. LLM fallback — only when rules are inconclusive.

This keeps the classifier reliable and cheap.
"""

import re
from typing import Literal

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Intent = Literal["greeting", "product_inquiry", "high_intent", "unknown"]

# ---------------------------------------------------------------------------
# Keyword banks (extend freely — order matters for precedence)
# ---------------------------------------------------------------------------

HIGH_INTENT_PATTERNS = [
    r"\bsign[\s-]?up\b",
    r"\bget started\b",
    r"\bi('m| am) (interested|ready|in)\b",
    r"\bwant to (try|buy|purchase|subscribe|join|start)\b",
    r"\bhow do i (sign|get|start|join|buy|subscribe)\b",
    r"\bpurchase\b",
    r"\bsubscribe\b",
    r"\bpro plan\b.*\b(want|try|get|buy)\b",
    r"\b(want|try|get|buy)\b.*\bpro plan\b",
    r"\bwhere (do i|can i) (sign|get|start|buy)\b",
    r"\blet('s| us) (do it|go|start|sign)\b",
    r"\bcount me in\b",
    r"\bready to (start|go|buy|pay)\b",
    r"\btake my (money|card|payment)\b",
]

PRODUCT_INQUIRY_PATTERNS = [
    r"\bpric(e|ing|es)\b",
    r"\bplan(s)?\b",
    r"\bfeature(s)?\b",
    r"\bhow (much|many|does|do)\b",
    r"\bwhat (is|are|does)\b",
    r"\btell me (about|more)\b",
    r"\brefund\b",
    r"\bsupport\b",
    r"\bcancel\b",
    r"\bupgrade\b",
    r"\btrial\b",
    r"\bbasic\b",
    r"\bpro\b",
    r"\b4k\b",
    r"\bvideo(s)?\b",
    r"\bcaption(s)?\b",
    r"\bplatform(s)?\b",
    r"\bpolic(y|ies)\b",
    r"\bcompare\b",
    r"\bdifference\b",
    r"\bwhat('s| is) included\b",
]

GREETING_PATTERNS = [
    r"^hi+\b",
    r"^hey+\b",
    r"^hello+\b",
    r"^howdy\b",
    r"^good (morning|afternoon|evening|day)\b",
    r"^what'?s up\b",
    r"^greetings\b",
    r"^yo\b",
    r"^sup\b",
]


def _match_any(text: str, patterns: list[str]) -> bool:
    """Return True if any pattern matches anywhere in text."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def classify_intent(user_message: str) -> Intent:
    """
    Classify a user message into one of: greeting, product_inquiry, high_intent, unknown.

    Precedence (highest → lowest):
      1. high_intent — user wants to take action
      2. product_inquiry — user has a question about product/pricing
      3. greeting — pure hello/hi
      4. unknown — fallback
    """
    text = user_message.strip()

    # High intent takes absolute precedence
    if _match_any(text, HIGH_INTENT_PATTERNS):
        return "high_intent"

    # Product/pricing inquiry
    if _match_any(text, PRODUCT_INQUIRY_PATTERNS):
        return "product_inquiry"

    # Greeting (only if message is short and matches — avoids misclassifying
    # "Hi, what's your pricing?" as a pure greeting)
    word_count = len(text.split())
    if word_count <= 6 and _match_any(text, GREETING_PATTERNS):
        return "greeting"

    # Short message that starts with a greeting but also has content
    if _match_any(text, GREETING_PATTERNS):
        return "product_inquiry"  # treat as product inquiry with greeting prefix

    return "unknown"
