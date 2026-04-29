"""
Locate run outputs under a session directory (nested ``<slug>/``, ``result/``, etc.).

Used by the web worker (enrich ``web_result.json``) and by FastAPI file/zip routes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def _is_safe_basename(name: str) -> bool:
    return bool(name) and ".." not in name and "/" not in name and "\\" not in name


def session_tree_files(workdir: Path, basename: str) -> List[Path]:
    """All files named ``basename`` under ``workdir`` (resolved, no escape)."""
    out: List[Path] = []
    if not workdir.is_dir() or not _is_safe_basename(basename):
        return out
    try:
        root_r = workdir.resolve()
    except OSError:
        return out
    try:
        for p in workdir.rglob(basename):
            if not p.is_file():
                continue
            try:
                p.resolve().relative_to(root_r)
            except ValueError:
                continue
            out.append(p)
    except OSError:
        pass
    return out


def pick_artifact_file(workdir: Path, basename: str) -> Optional[Path]:
    """Pick one file: prefer curated result folders, then newest mtime."""
    cands = session_tree_files(workdir, basename)
    if not cands:
        return None

    def sort_key(p: Path) -> tuple:
        if "mlip_results" in p.parts:
            bucket = 0
        elif "vasp_results" in p.parts:
            bucket = 1
        elif "result" in p.parts:
            bucket = 2
        else:
            bucket = 3
        try:
            mt = -p.stat().st_mtime_ns
        except OSError:
            mt = 0
        return (bucket, mt)

    cands.sort(key=sort_key)
    return cands[0]


def find_pairs_best_it_dir(workdir: Path) -> Optional[Path]:
    """Directory named ``pairs_best_it`` under the session tree."""
    if not workdir.is_dir():
        return None
    try:
        root_r = workdir.resolve()
    except OSError:
        return None
    found: List[Path] = []
    try:
        for p in workdir.rglob("pairs_best_it"):
            if not p.is_dir() or p.name != "pairs_best_it":
                continue
            try:
                p.resolve().relative_to(root_r)
            except ValueError:
                continue
            found.append(p)
    except OSError:
        return None
    if not found:
        return None

    def sort_key(p: Path) -> tuple:
        in_result = 0 if "result" in p.parts else 1
        try:
            mt = -p.stat().st_mtime_ns
        except OSError:
            mt = 0
        return (in_result, mt)

    found.sort(key=sort_key)
    return found[0]


def enrich_ok_payload_artifacts(workdir: Path, payload: Dict[str, Any]) -> None:
    """
    Fill missing ``artifacts`` paths by scanning the session tree.

    Sets ``artifacts_notes`` when interactive stereographic HTML is still missing
    (usually ``plotly`` not installed).
    """
    if not payload.get("ok"):
        return
    art = payload.get("artifacts")
    if not isinstance(art, dict):
        art = {}
        payload["artifacts"] = art

    mapping = (
        ("selected_interfaces.csv", "selected_interfaces_csv"),
        ("all_match_info", "all_match_info"),
        ("stereographic_interactive.html", "stereographic_interactive_html"),
        ("stereographic.jpg", "stereographic_jpg"),
    )
    for fname, key in mapping:
        cur = art.get(key)
        if isinstance(cur, str) and cur.strip():
            try:
                if Path(cur).expanduser().is_file():
                    continue
            except OSError:
                pass
        picked = pick_artifact_file(workdir, fname)
        if picked is not None:
            art[key] = str(picked)
    if not art.get("all_match_info"):
        picked = pick_artifact_file(workdir, "area_match")
        if picked is not None:
            art["all_match_info"] = str(picked)

    if not art.get("stereographic_interactive_html"):
        notes = payload.get("artifacts_notes")
        if not isinstance(notes, dict):
            notes = {}
            payload["artifacts_notes"] = notes
        notes["stereographic_interactive"] = (
            "未找到 stereographic_interactive.html。请在服务器执行: pip install plotly 后重新计算。"
        )
