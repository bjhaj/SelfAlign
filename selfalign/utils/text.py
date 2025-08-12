from __future__ import annotations

import re
from typing import Iterable


def simple_tokenize(text: str) -> list[str]:
    """Very simple whitespace/punctuation tokenizer (stub)."""
    return re.findall(r"\w+", text.lower())


def has_any(text: str, needles: Iterable[str]) -> bool:
    """Return True if any needle substring occurs in text (stub)."""
    t = text.lower()
    return any(n.lower() in t for n in needles)


# --- New helpers ---

def to_kebab_case(s: str) -> str:
    """Convert arbitrary strings to kebab-case.

    Behavior:
    - Lowercase
    - Spaces/underscores and any non-alphanumeric sequences → single hyphen
    - Collapse repeated hyphens
    - Strip leading/trailing hyphens

    Examples:
    >>> to_kebab_case(" Next Steps ")
    'next-steps'
    >>> to_kebab_case("Claim_Evidence Counterpoint")
    'claim-evidence-counterpoint'
    >>> to_kebab_case("--Weird__Name!!//here")
    'weird-name-here'
    """
    s = (s or "").strip().lower()
    # Replace spaces/underscores and any non-alnum with hyphen
    s = re.sub(r"[^a-z0-9]+", "-", s)
    # Collapse multiple hyphens
    s = re.sub(r"-{2,}", "-", s)
    # Trim hyphens
    return s.strip("-")


def split_tone(s: str) -> list[str]:
    """Split a comma-separated tone string into unique, lowercased tokens.

    Behavior:
    - Split on commas
    - Trim whitespace, lowercase
    - Drop empty tokens
    - Dedupe while preserving order

    Examples:
    >>> split_tone("Calm, Precise, inquisitive")
    ['calm', 'precise', 'inquisitive']
    >>> split_tone("calm, calm, precise ,,  ")
    ['calm', 'precise']
    """
    seen: set[str] = set()
    out: list[str] = []
    for part in (s or "").split(","):
        tok = part.strip().lower()
        if not tok:
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out
