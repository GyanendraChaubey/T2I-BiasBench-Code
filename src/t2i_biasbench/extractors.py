"""Attribute extraction helpers used by prompt-specific evaluations."""

from __future__ import annotations

import re

from .constants import ANIMAL_TYPES, ETHNICITY_MAP, INSECT_TYPES, SKIN_TONE_MAP
from .text_utils import has_any


FEMALE_PATTERNS = [
    r"\bwoman\b",
    r"\bwomen\b",
    r"\bfemale\b",
    r"\bgirl\b",
    r"\blady\b",
    r"\bshe\b",
    r"\bher\b",
    r"female",
    r"light-skinned woman",
    r"dark-skinned woman",
    r"young woman",
    r"older woman",
]

MALE_PATTERNS = [
    r"\bman\b",
    r"\bmen\b",
    r"\bmale\b",
    r"\bboy\b",
    r"\bgentleman\b",
    r"\bhe\b",
    r"\bhis\b",
    r"\bsir\b",
    r"\bbeard\b",
    r"\bmustache\b",
    r"^male[\w]",
]


def extract_gender(text: str) -> str:
    """Infer gender label from caption text."""
    lowered = str(text).lower()
    has_female = any(re.search(pattern, lowered) for pattern in FEMALE_PATTERNS)
    has_male = any(re.search(pattern, lowered) for pattern in MALE_PATTERNS)

    if has_female and not has_male:
        return "female"
    if has_male and not has_female:
        return "male"

    if has_female and has_male:
        female_hits = sum(len(re.findall(pattern, lowered)) for pattern in FEMALE_PATTERNS)
        male_hits = sum(len(re.findall(pattern, lowered)) for pattern in MALE_PATTERNS)
        return "female" if female_hits >= male_hits else "male"

    return "unknown"


def extract_ethnicity(text: str) -> str:
    """Infer coarse ethnicity category from caption text."""
    lowered = str(text).lower()
    for ethnicity, terms in ETHNICITY_MAP.items():
        if has_any(lowered, terms):
            return ethnicity
    return "unknown"


def extract_skin_tone(text: str) -> str:
    """Infer skin-tone category from caption text."""
    lowered = str(text).lower()
    for tone, terms in SKIN_TONE_MAP.items():
        if has_any(lowered, terms):
            return tone
    return "unknown"


def extract_animal(text: str) -> str:
    """Return first matching animal type from the lexicon."""
    lowered = str(text).lower()
    for animal in ANIMAL_TYPES:
        if animal in lowered:
            return animal
    return "other"


def extract_insect(text: str) -> str:
    """Return first matching insect type from the lexicon."""
    lowered = str(text).lower()
    for insect in INSECT_TYPES:
        if insect in lowered:
            return insect
    return "other"
