"""
Optional runtime visualization hooks for the desktop GUI.

Visualization is **off** unless both are set:

- ``INTEROPTIMUS_VIZ_LOG`` — path to a JSONL file
- ``INTEROPTIMUS_VIZ_ENABLE`` — ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive)

Normal workflows (CLI, tests, default GUI) do not set these, so no overhead.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict

_lock = threading.Lock()


def is_enabled() -> bool:
    """True only when log path is set and explicit enable flag is on."""
    path = (os.environ.get("INTEROPTIMUS_VIZ_LOG") or "").strip()
    if not path:
        return False
    flag = str(os.environ.get("INTEROPTIMUS_VIZ_ENABLE", "")).strip().lower()
    return flag in ("1", "true", "yes", "on")


def emit_event(payload: Dict[str, Any]) -> None:
    if not is_enabled():
        return
    path = (os.environ.get("INTEROPTIMUS_VIZ_LOG") or "").strip()
    if not path:
        return
    row = dict(payload)
    row.setdefault("v", 1)
    line = json.dumps(row, default=str, ensure_ascii=False) + "\n"
    try:
        with _lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except OSError:
        pass
