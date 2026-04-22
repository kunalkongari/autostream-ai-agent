"""
rag.py — Retrieval-Augmented Generation pipeline.

Approach: Keyword-based retrieval over a local JSON knowledge base.
  - No vector DB required (keeps things simple and fast).
  - Retrieves the most relevant sections based on keyword overlap.
  - Returns a formatted context string to be injected into the LLM prompt.

If you need semantic search later, swap retrieve() to use sentence-transformers
or any embedding model — the interface stays the same.
"""

import json
import re
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Load knowledge base at import time (cheap — tiny JSON file)
# ---------------------------------------------------------------------------

_KB_PATH = Path(__file__).parent.parent / "data" / "knowledge_base.json"

with open(_KB_PATH, "r", encoding="utf-8") as f:
    _KB: dict = json.load(f)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _keyword_score(text: str, keywords: List[str]) -> int:
    """Count how many keywords appear in text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def _format_plan(plan: dict) -> str:
    features = "\n".join(f"    • {f}" for f in plan["features"])
    return (
        f"**{plan['name']}** — {plan['price']}\n"
        f"  Features:\n{features}"
    )


def _format_policy(policy: dict) -> str:
    return f"**{policy['topic']}**: {policy['detail']}"


def _format_faq(faq: dict) -> str:
    return f"**Q: {faq['question']}**\n  A: {faq['answer']}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = 3) -> str:
    """
    Given a user query, retrieve the most relevant knowledge base entries
    and return them as a formatted context string.

    Args:
        query: The raw user message.
        top_k: Maximum number of sections to include.

    Returns:
        A formatted string of retrieved context, or a fallback message.
    """
    candidates: List[Tuple[int, str]] = []  # (score, formatted_text)

    # Score each plan
    for plan in _KB.get("plans", []):
        score = _keyword_score(query, plan.get("keywords", []))
        # Always include plans if query mentions pricing-related terms
        if re.search(r"\bpric|plan|cost|fee|cheap|afford|basic|pro\b", query, re.I):
            score = max(score, 1)
        if score > 0:
            candidates.append((score, _format_plan(plan)))

    # Score each policy
    for policy in _KB.get("policies", []):
        score = _keyword_score(query, policy.get("keywords", []))
        if score > 0:
            candidates.append((score, _format_policy(policy)))

    # Score each FAQ entry
    for faq in _KB.get("faq", []):
        score = _keyword_score(query, faq.get("keywords", []))
        if score > 0:
            candidates.append((score, _format_faq(faq)))

    if not candidates:
        # If no match, return full pricing overview as a sensible default
        return _full_pricing_context()

    # Sort by score descending, take top_k
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = [text for _, text in candidates[:top_k]]

    return "\n\n".join(selected)


def _full_pricing_context() -> str:
    """Return the complete pricing table as a fallback."""
    plans_text = "\n\n".join(_format_plan(p) for p in _KB.get("plans", []))
    return (
        f"AutoStream Pricing Overview:\n\n{plans_text}\n\n"
        "Policies:\n"
        + "\n".join(_format_policy(p) for p in _KB.get("policies", []))
    )


def get_company_intro() -> str:
    """Return a short company description for greeting responses."""
    c = _KB.get("company", {})
    return c.get("description", "AutoStream is an AI-powered video editing SaaS for content creators.")
