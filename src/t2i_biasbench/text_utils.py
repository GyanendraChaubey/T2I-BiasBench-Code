"""Text normalization and prompt classification helpers."""

from __future__ import annotations

import re
from typing import Mapping, Sequence


def clean_text(text: str) -> str:
    """Normalize caption text for robust term matching."""
    normalized = str(text).lower()
    normalized = normalized.replace("/", " ").replace("-", " ").replace(";", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def has_any(text: str, terms: Sequence[str]) -> bool:
    """Return True if any term appears as a substring in text."""
    lowered = str(text).lower()
    return any(term in lowered for term in terms)


def detect_prompt(image_name: str, prompt_rules: Mapping[str, Sequence[str]]) -> str:
    """Map an image name to a prompt bucket using ordered keyword rules."""
    name = str(image_name).lower()
    for prompt, keywords in prompt_rules.items():
        if any(keyword.lower() in name for keyword in keywords):
            return prompt
    return "unknown"
