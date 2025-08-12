from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def _short_sha() -> str:
    return "local"


def new_run_dir(prefix: str = "") -> str:
    """Create runs/<timestamp>_<shortsha> and return its path."""
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    name = f"{prefix + '_' if prefix else ''}{ts}_{_short_sha()}"
    run_dir = Path("runs") / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def log_event(run_dir: str, event: str, payload: dict[str, Any]) -> None:
    """Append JSONL event to run log (stub)."""
    path = Path(run_dir) / "events.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event, **payload}) + "\n")


def write_json(path: str, obj: dict) -> None:
    """Write obj as JSON to path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
