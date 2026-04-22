"""
Small helpers shared with :mod:`InterOptimus.session_workflow` and config tooling.

The full IOMaker pipeline lives in :mod:`InterOptimus.agents.iomaker_job`.
"""

from __future__ import annotations

from typing import Optional


def _normalize_mlip_calc(name: Optional[str]) -> Optional[str]:
    """
    Map user-facing MLIP backend strings to ``InterfaceWorker`` / ``MLIPCalculator`` names.

    Returns ``None`` if *name* is empty or not recognized (caller supplies a default).
    """
    if not name or not isinstance(name, str):
        return None
    s = name.strip().lower().replace("_", "-")
    aliases = {
        "orb": "orb-models",
        "orb_models": "orb-models",
        "7net": "sevenn",
        "deepmd": "dpa",
        "deep-potential": "dpa",
    }
    if s in aliases:
        return aliases[s]
    if s in ("orb-models", "sevenn", "matris", "dpa"):
        return s
    return None
