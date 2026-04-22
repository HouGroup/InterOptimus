"""
Optional runtime visualization hooks (JSONL event stream for the web UI).

Visualization is **off** unless both are set:

- ``INTEROPTIMUS_VIZ_LOG`` — path to a JSONL file
- ``INTEROPTIMUS_VIZ_ENABLE`` — ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive)

Normal CLI runs omit these, so there is no overhead.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any, Dict, Optional

_lock = threading.Lock()
# Set by :func:`pin_web_viz_session` (web worker) so logging still works if ``os.environ`` is
# mutated mid-run; optional defensive guard only.
_forced_log_path: Optional[str] = None


def pin_web_viz_session(path: str) -> None:
    """
    Pin the JSONL path for this process and sync env (used by ``web_app.job_worker``).

    Keeps ``INTEROPTIMUS_VIZ_*`` consistent with the pinned path even if user code touches
    the environment.
    """
    global _forced_log_path
    _forced_log_path = str(path).strip()
    if _forced_log_path:
        os.environ["INTEROPTIMUS_VIZ_LOG"] = _forced_log_path
        os.environ["INTEROPTIMUS_VIZ_ENABLE"] = "1"


def is_enabled() -> bool:
    """True only when log path is set and explicit enable flag is on."""
    path = (_forced_log_path or os.environ.get("INTEROPTIMUS_VIZ_LOG") or "").strip()
    if not path:
        return False
    flag = str(os.environ.get("INTEROPTIMUS_VIZ_ENABLE", "")).strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    # Web worker pinned path but something cleared the enable flag — still stream JSONL.
    return bool(_forced_log_path) and path == _forced_log_path


def emit_event(payload: Dict[str, Any]) -> None:
    if not is_enabled():
        return
    path = (_forced_log_path or os.environ.get("INTEROPTIMUS_VIZ_LOG") or "").strip()
    if not path:
        return
    row = dict(payload)
    row.setdefault("v", 1)
    line = json.dumps(row, default=str, ensure_ascii=False) + "\n"
    try:
        with _lock:
            # Line-buffered append so clients tailing this file see events immediately.
            with open(path, "a", encoding="utf-8", buffering=1) as f:
                f.write(line)
                f.flush()
    except OSError as e:
        try:
            print(f"[InterOptimus] viz emit_event OSError ({path}): {e}", file=sys.stderr, flush=True)
        except OSError:
            pass
