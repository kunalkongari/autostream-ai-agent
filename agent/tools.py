"""
tools.py — Tool execution layer.

Contains the mock lead capture function and the wrapper that validates
state before calling it. The agent node must call `execute_lead_capture()`
rather than `mock_lead_capture()` directly — this enforces the guard.
"""

import re
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Mock lead capture (as specified in the assignment)
# ---------------------------------------------------------------------------

def mock_lead_capture(name: str, email: str, platform: str) -> None:
    """
    Simulates a CRM / backend API call that registers a new lead.
    In production this would POST to a CRM endpoint (HubSpot, Salesforce, etc.).
    """
    print(f"\n{'='*50}")
    print(f"✅  LEAD CAPTURED SUCCESSFULLY")
    print(f"    Name     : {name}")
    print(f"    Email    : {email}")
    print(f"    Platform : {platform}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _is_valid_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email.strip()))


def _is_valid_name(name: str) -> bool:
    return len(name.strip()) >= 2


KNOWN_PLATFORMS = {
    "youtube", "instagram", "tiktok", "twitter", "x",
    "linkedin", "facebook", "twitch", "snapchat", "pinterest",
}

def _is_valid_platform(platform: str) -> bool:
    return len(platform.strip()) >= 2  # accept any non-trivial string


# ---------------------------------------------------------------------------
# Safe executor with pre-call validation
# ---------------------------------------------------------------------------

def execute_lead_capture(
    name: Optional[str],
    email: Optional[str],
    platform: Optional[str],
) -> Tuple[bool, str]:
    """
    Validates all fields, then calls mock_lead_capture().

    Returns:
        (success: bool, message: str)
        - success=True if lead was captured.
        - message describes the outcome or the validation error.
    """
    # Guard: all fields must be present
    if not name or not email or not platform:
        missing = [f for f, v in [("name", name), ("email", email), ("platform", platform)] if not v]
        return False, f"Cannot capture lead — missing fields: {', '.join(missing)}"

    # Validate each field
    if not _is_valid_name(name):
        return False, "Name appears invalid. Please provide your full name."

    if not _is_valid_email(email):
        return False, f"'{email}' doesn't look like a valid email address. Please double-check."

    if not _is_valid_platform(platform):
        return False, "Platform name is too short. Please specify (e.g. YouTube, Instagram)."

    # All good — call the mock API
    mock_lead_capture(name.strip(), email.strip(), platform.strip())
    return True, "Lead captured successfully."


# ---------------------------------------------------------------------------
# Field extraction helpers (parse user replies during collection stage)
# ---------------------------------------------------------------------------

def extract_email_from_text(text: str) -> Optional[str]:
    """Pull the first email-like token from a user message."""
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else None


def extract_platform_from_text(text: str) -> Optional[str]:
    """Detect a known platform name in the user message."""
    text_lower = text.lower()
    for platform in KNOWN_PLATFORMS:
        if platform in text_lower:
            return platform.capitalize()
    # If no known platform found, return the whole (stripped) message as the platform
    cleaned = text.strip().split()[0] if text.strip() else None
    return cleaned
