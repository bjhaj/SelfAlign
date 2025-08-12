from __future__ import annotations

from typing import Iterable, Tuple, List, Optional, Dict
from pydantic import BaseModel, Field
import re
import json
import hashlib
from pathlib import Path

from selfalign.utils.io import load_yaml_dict

# ---------------------------------------------------------------------------
# Constants and regex patterns used across persona validation
# ---------------------------------------------------------------------------
# Fixed ordering of persona value dimensions (0..1 expected for each)
ORDERED_VALUES: list[str] = [
    "candor",
    "prudence",
    "helpfulness",
    "skepticism",
    "risk_aversion",
    "empathy",
    "neutrality",
]

# ID must be kebab-case, version must be semver
KEBAB_RE: str = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
SEMVER_RE: str = r"^\d+\.\d+\.\d+$"

# Tone must avoid vague adjectives (case-insensitive check should be applied)
BANNED_TONE_WORDS: set[str] = {"good", "nice", "friendly", "awesome", "cool"}

# Module-level warning collector for last validation pass
last_warnings: List[str] = []


# ---------------------------------------------------------------------------
# Typed models (Pydantic) describing the persona schema
# Note: Validation/normalization to be added in later iterations (TODOs)
# ---------------------------------------------------------------------------
class PersonaStyle(BaseModel):
    """Style configuration for a persona.

    Fields:
        tone: A short, comma-separated list of adjectives (e.g., "calm, precise").
              Should avoid banned/vague words and be <= ~6 descriptors. (TODO)
        verbosity: One of: short | medium | long. (TODO)
        structure: Ordered list (0–6 items) of section names; prefer kebab-case. (TODO)
    """

    tone: str = Field(..., description="Comma-separated adjectives describing tone.")
    verbosity: str = Field(..., description="Verbosity level: short|medium|long.")
    structure: list[str] = Field(default_factory=list, description="Ordered list of section names (0–6).")


class RefusalPolicy(BaseModel):
    """Refusal policy for the persona.

    Fields:
        band: Target refusal rate range [low, high], each in [0,1], low ≤ high. (TODO)
        template: Surface text for refusals. (TODO)
    """

    band: list[float] = Field(..., description="Target refusal band [low, high] in [0,1].")
    template: str = Field(..., description="Refusal surface text template.")


class Redlines(BaseModel):
    """Redlines defining safety/privacy constraints.

    Fields:
        safety: Non-empty list of safety topics/phrases. (TODO)
        privacy: Non-empty list of privacy topics/phrases. (TODO)
    """

    safety: list[str] = Field(default_factory=list, description="Safety redlines list.")
    privacy: list[str] = Field(default_factory=list, description="Privacy redlines list.")


class Persona(BaseModel):
    """Persona specification.

    Fields:
        id: Kebab-case identifier (e.g., "socratic-skeptical"). (TODO validate via KEBAB_RE)
        version: Semantic version string (e.g., "0.1.0"). (TODO validate via SEMVER_RE)
        description: Short description (<= ~180 chars recommended). (TODO)
        values: Mapping from ORDERED_VALUES keys to floats in [0,1]. (TODO full check)
        style: PersonaStyle block (tone/verbosity/structure).
        taboos: Non-empty list of strings; content-level taboos. (TODO)
        redlines: Redlines block (safety/privacy lists).
        tie_breaks: Optional list of heuristics/notes.
        refusal_policy: RefusalPolicy block (band/template).
    """

    id: str = Field(..., description="Kebab-case persona id.")
    version: str = Field(..., description="Semantic version.")
    description: str = Field(..., description="Short description of the persona.")
    values: Dict[str, float] = Field(default_factory=dict, description="Persona value weights in [0,1].")
    style: PersonaStyle
    taboos: list[str] = Field(default_factory=list)
    redlines: Redlines
    tie_breaks: Optional[list[str]] = None
    refusal_policy: RefusalPolicy

    # TODO: Add model-level validation to enforce:
    # - id matches KEBAB_RE
    # - version matches SEMVER_RE
    # - values have all ORDERED_VALUES keys and each in [0,1]
    # - tone bans BANNED_TONE_WORDS (case-insensitive)
    # - verbosity ∈ {short, medium, long}
    # - 0 ≤ len(structure) ≤ 6 and prefer kebab-case for names
    # - non-empty lists for taboos, redlines.safety, redlines.privacy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Helper to collect errors with a JSONPointer-like path
_def_path_root = "/"

def _collect_error(errors: list[str], path: str, msg: str) -> None:
    pointer = path if path.startswith("/") else ("/" + path)
    errors.append(f"ERROR: {pointer}: {msg}")


def _to_kebab(s: str) -> str:
    s2 = re.sub(r"[^A-Za-z0-9]+", "-", s.strip().lower())
    s2 = re.sub(r"-+", "-", s2)
    return s2.strip("-")


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def compute_style_centroid_placeholder(persona: dict) -> list[float]:
    """Compute a deterministic placeholder style centroid vector.

    Uses a simple hashing scheme over tone_tags + structure into a length-16
    bucketed vector, then L1 normalizes counts. This is a placeholder for
    real embeddings and exists to unblock downstream code paths.
    """
    tokens: list[str] = []
    style = persona.get("style", {}) if isinstance(persona, dict) else {}
    tokens.extend(style.get("tone_tags", []) or [])
    tokens.extend(style.get("structure", []) or [])

    size = 16
    vec = [0] * size
    total = 0
    for tok in tokens:
        if not tok:
            continue
        key = str(tok).strip().lower()
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % size
        vec[idx] += 1
        total += 1
    if total > 0:
        vec = [v / float(total) for v in vec]
    else:
        vec = [0.0 for _ in vec]
    return vec


def _normalize_persona(raw: dict) -> dict:
    """Return canonical normalized persona dict.

    - Strip description.
    - Round values to 3 decimals in ORDERED_VALUES order.
    - style.tone_tags: comma-split, trim, lowercase, dedupe.
    - style.verbosity: lowercase to one of {short, medium, long}.
    - style.structure: kebab-case entries; dedupe preserving order.
    - refusal_policy.band: round to 3 decimals; template strip + collapse spaces.
    - Add values_vector in ORDERED_VALUES order (floats).
    - Add style_centroid_placeholder: length-16 vector derived from style tokens.
    - Add schema_signature: sha256 prefix over the normalized dict (first 8 hex).
    - Return only canonical fields.
    """
    pid = str(raw["id"]).strip()
    ver = str(raw["version"]).strip()
    desc = str(raw["description"]).strip()

    # values mapping and vector (rounded)
    vals_map: dict[str, float] = {}
    vals_vec: list[float] = []
    for k in ORDERED_VALUES:
        v = float(raw["values"][k])
        vr = round(v, 3)
        vals_map[k] = vr
        vals_vec.append(vr)

    # style
    style_raw = raw["style"]
    tone = style_raw.get("tone", "")
    parts = [p.strip().lower() for p in str(tone).split(",") if p.strip()]
    tone_tags = _dedupe_preserve_order(parts)

    verbosity = str(style_raw.get("verbosity", "")).lower().strip()
    if verbosity not in {"short", "medium", "long"}:
        verbosity = "medium"

    structure_raw = style_raw.get("structure", []) or []
    structure_kebab = [_to_kebab(str(s)) for s in structure_raw]
    structure_kebab = [s for s in structure_kebab if s]
    structure_kebab = _dedupe_preserve_order(structure_kebab)

    # redlines
    red = raw["redlines"]
    safety = [str(x).strip() for x in red.get("safety", []) if str(x).strip()]
    privacy = [str(x).strip() for x in red.get("privacy", []) if str(x).strip()]

    # taboos
    taboos = [str(x).strip() for x in raw.get("taboos", []) if str(x).strip()]

    # tie_breaks (optional)
    tie_breaks = None
    if raw.get("tie_breaks") is not None:
        tbs = [str(x).strip() for x in raw.get("tie_breaks", []) if str(x).strip()]
        tie_breaks = tbs

    # refusal policy
    rp = raw["refusal_policy"]
    band = [round(float(rp["band"][0]), 3), round(float(rp["band"][1]), 3)]
    templ = " ".join(str(rp.get("template", "")).strip().split())

    canonical = {
        "id": pid,
        "version": ver,
        "description": desc,
        "values": {k: vals_map[k] for k in ORDERED_VALUES},
        "style": {
            "tone_tags": tone_tags,
            "verbosity": verbosity,
            "structure": structure_kebab,
        },
        "taboos": taboos,
        "redlines": {
            "safety": safety,
            "privacy": privacy,
        },
        "refusal_policy": {
            "band": band,
            "template": templ,
        },
        "values_vector": vals_vec,
    }
    if tie_breaks is not None:
        canonical["tie_breaks"] = tie_breaks

    # Add placeholder centroid before computing signature
    canonical["style_centroid_placeholder"] = compute_style_centroid_placeholder(canonical)

    # Make stable signature over canonical content
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    sig = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    canonical["schema_signature"] = sig
    return canonical


def validate_persona_yaml(path: str) -> dict:
    """Load + validate persona YAML against fixed ontology; raise ValueError on errors.

    Args:
        path: Path to persona YAML file.
    Returns:
        Canonical, normalized persona configuration (as dict).
    Raises:
        ValueError: Aggregated errors if validation fails.
    """
    data = load_yaml_dict(path)
    errors: list[str] = []
    warnings: list[str] = []

    # Top-level keys: required and allowed
    allowed_top = {
        "id",
        "version",
        "description",
        "values",
        "style",
        "taboos",
        "redlines",
        "tie_breaks",
        "refusal_policy",
        # canonical-only optional fields for roundtrip
        "values_vector",
        "schema_signature",
        "style_centroid_placeholder",
    }
    required_top = {"id", "version", "description", "values", "style", "taboos", "redlines", "refusal_policy"}

    # Unknown keys
    unknown = sorted(k for k in data.keys() if k not in allowed_top)
    for k in unknown:
        _collect_error(errors, f"{k}", "Unknown key")

    # Missing required keys
    missing = sorted(k for k in required_top if k not in data)
    for k in missing:
        _collect_error(errors, f"{k}", "Missing required key")

    # id
    if "id" in data:
        if not isinstance(data["id"], str):
            _collect_error(errors, "id", "Must be a string")
        else:
            if not re.match(KEBAB_RE, data["id"]):
                _collect_error(errors, "id", "Must be kebab-case (e.g., socratic-skeptical)")

    # version
    if "version" in data:
        if not isinstance(data["version"], str):
            _collect_error(errors, "version", "Must be a string")
        else:
            if not re.match(SEMVER_RE, data["version"]):
                _collect_error(errors, "version", "Must be semantic version (e.g., 0.1.0)")

    # description
    if "description" in data:
        if not isinstance(data["description"], str):
            _collect_error(errors, "description", "Must be a string")
        else:
            if len(data["description"]) > 180:
                _collect_error(errors, "description", "Must be ≤ 180 characters")

    # values
    if "values" in data:
        vals = data["values"]
        if not isinstance(vals, dict):
            _collect_error(errors, "values", "Must be a mapping of value_name → float [0,1]")
        else:
            for key in ORDERED_VALUES:
                if key not in vals:
                    _collect_error(errors, f"values/{key}", "is missing")
                else:
                    v = vals[key]
                    if not isinstance(v, (int, float)):
                        _collect_error(errors, f"values/{key}", "Must be a number in [0,1]")
                    else:
                        vf = float(v)
                        if not (0.0 <= vf <= 1.0):
                            _collect_error(errors, f"values/{key}", "Must be within [0,1]")

    # style
    if "style" in data:
        st = data["style"]
        if not isinstance(st, dict):
            _collect_error(errors, "style", "Must be a mapping with tone/tone_tags, verbosity, structure")
        else:
            # tone or tone_tags (canonical)
            has_tone = "tone" in st
            has_tags = isinstance(st.get("tone_tags"), list)
            if not has_tone and not has_tags:
                _collect_error(errors, "style/tone", "Missing required key (tone or tone_tags)")
            else:
                if has_tone:
                    tone = st["tone"]
                    if not isinstance(tone, str):
                        _collect_error(errors, "style/tone", "Must be a string of comma-separated descriptors")
                    else:
                        descriptors = [p.strip() for p in tone.split(",")]
                        if len(descriptors) > 6:
                            _collect_error(errors, "style/tone", "Must have ≤ 6 descriptors (comma-separated)")
                        for d in descriptors:
                            for tok in re.findall(r"[A-Za-z0-9]+", d.lower()):
                                if tok in BANNED_TONE_WORDS:
                                    _collect_error(errors, "style/tone", f"Banned vague descriptor: '{tok}'")
                if has_tags:
                    tags_list = st.get("tone_tags") or []
                    if not isinstance(tags_list, list):
                        _collect_error(errors, "style/tone_tags", "Must be a list of strings")
                    else:
                        if len(tags_list) > 6:
                            _collect_error(errors, "style/tone_tags", "Must have ≤ 6 descriptors")
                        for i, tag in enumerate(tags_list):
                            if not isinstance(tag, str) or not tag.strip():
                                _collect_error(errors, f"style/tone_tags/{i}", "Must be a non-empty string")
                            else:
                                tok = re.sub(r"[^A-Za-z0-9]+", "", tag.lower())
                                if tok in BANNED_TONE_WORDS:
                                    _collect_error(errors, "style/tone_tags", f"Banned vague descriptor: '{tok}'")
            # verbosity
            if "verbosity" not in st:
                _collect_error(errors, "style/verbosity", "Missing required key")
            else:
                vb = st["verbosity"]
                if not isinstance(vb, str):
                    _collect_error(errors, "style/verbosity", "Must be a string: short|medium|long")
                else:
                    if vb.lower() not in {"short", "medium", "long"}:
                        _collect_error(errors, "style/verbosity", "Must be one of: short, medium, long (case-insensitive)")
            # structure
            if "structure" not in st:
                _collect_error(errors, "style/structure", "Missing required key")
            else:
                struct = st["structure"]
                if not isinstance(struct, list):
                    _collect_error(errors, "style/structure", "Must be a list of section names (kebab-case)")
                else:
                    if not (0 <= len(struct) <= 6):
                        _collect_error(errors, "style/structure", "Length must be between 0 and 6")
                    for i, item in enumerate(struct):
                        ipath = f"style/structure/{i}"
                        if not isinstance(item, str):
                            _collect_error(errors, ipath, "Must be a string")
                            continue
                        if item.strip() == "":
                            _collect_error(errors, ipath, "Must not be empty")
                        elif not re.match(KEBAB_RE, item):
                            pointer = ipath if ipath.startswith("/") else ("/" + ipath)
                            warnings.append(f"WARN: {pointer}: not kebab-case; will normalize")

    # refusal_policy
    if "refusal_policy" in data:
        rp = data["refusal_policy"]
        if not isinstance(rp, dict):
            _collect_error(errors, "refusal_policy", "Must be a mapping with band and template")
        else:
            # band
            if "band" not in rp:
                _collect_error(errors, "refusal_policy/band", "Missing required key")
            else:
                band = rp["band"]
                if not (isinstance(band, list) and len(band) == 2 and all(isinstance(x, (int, float)) for x in band)):
                    _collect_error(errors, "refusal_policy/band", "Must be a list [low, high] with numbers")
                else:
                    low, high = float(band[0]), float(band[1])
                    if not (0.0 <= low <= high <= 1.0):
                        _collect_error(errors, "refusal_policy/band", "Require 0 ≤ low ≤ high ≤ 1")
            # template
            if "template" not in rp:
                _collect_error(errors, "refusal_policy/template", "Missing required key")
            else:
                t = rp["template"]
                if not isinstance(t, str):
                    _collect_error(errors, "refusal_policy/template", "Must be a string")
                else:
                    if t.strip() == "":
                        _collect_error(errors, "refusal_policy/template", "Must not be empty")
                    if len(t) > 200:
                        _collect_error(errors, "refusal_policy/template", "Must be ≤ 200 characters")

    # redlines
    if "redlines" in data:
        rl = data["redlines"]
        if not isinstance(rl, dict):
            _collect_error(errors, "redlines", "Must be a mapping with safety and privacy lists")
        else:
            for key in ("safety", "privacy"):
                if key not in rl:
                    _collect_error(errors, f"redlines/{key}", "Missing required key")
                    continue
                lst = rl[key]
                if not isinstance(lst, list):
                    _collect_error(errors, f"redlines/{key}", "Must be a list of strings")
                else:
                    if len(lst) == 0:
                        _collect_error(errors, f"redlines/{key}", "Must be non-empty")
                    for i, item in enumerate(lst):
                        ipath = f"redlines/{key}/{i}"
                        if not isinstance(item, str):
                            _collect_error(errors, ipath, "Must be a string")
                        elif item.strip() == "":
                            _collect_error(errors, ipath, "Must not be empty")

    # taboos
    if "taboos" in data:
        tb = data["taboos"]
        if not isinstance(tb, list):
            _collect_error(errors, "taboos", "Must be a list of strings")
        else:
            if len(tb) == 0:
                _collect_error(errors, "taboos", "Must be non-empty")
            for i, item in enumerate(tb):
                ipath = f"taboos/{i}"
                if not isinstance(item, str):
                    _collect_error(errors, ipath, "Must be a string")
                elif item.strip() == "":
                    _collect_error(errors, ipath, "Must not be empty")

    # tie_breaks (optional)
    if "tie_breaks" in data:
        tb = data["tie_breaks"]
        if not isinstance(tb, list):
            _collect_error(errors, "tie_breaks", "Must be a list of strings if provided")
        else:
            for i, item in enumerate(tb):
                ipath = f"tie_breaks/{i}"
                if not isinstance(item, str):
                    _collect_error(errors, ipath, "Must be a string")
                elif item.strip() == "":
                    _collect_error(errors, ipath, "Must not be empty")

    if errors:
        raise ValueError("\n".join(errors))

    # Normalize and collect filename warning
    canonical = _normalize_persona(data)

    # Reset and collect warnings for this validation pass
    last_warnings.clear()
    last_warnings.extend(warnings)
    try:
        base = Path(path).name
        expected = f"{canonical['id']}.yaml"
        if base != expected:
            last_warnings.append(f"WARN: /id: filename '{base}' does not match '<id>.yaml' ('{expected}')")
    except Exception:
        pass

    return canonical


def lint_persona_files(paths: Iterable[str]) -> List[Tuple[str, bool, List[str], List[str]]]:
    """Return (path, ok, errors, warnings) for each persona YAML.

    Args:
        paths: Iterable of file paths.
    Returns:
        List of tuples (path, ok, errors, warnings).
    """
    results: List[Tuple[str, bool, List[str], List[str]]] = []
    for p in paths:
        try:
            # Validate and capture warnings from the last pass
            validate_persona_yaml(p)
            warnings = list(last_warnings)
            results.append((p, True, [], warnings))
        except ValueError as e:
            errors = str(e).splitlines()
            results.append((p, False, errors, []))
        except Exception as e:
            results.append((p, False, [f"ERROR: /: unexpected exception: {e.__class__.__name__}: {e}"], []))
    return results
