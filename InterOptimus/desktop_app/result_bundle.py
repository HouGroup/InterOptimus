"""
After a successful desktop run, populate ``<run_dir>/result/`` with a minimal artifact set:

- ``results.csv`` (lattice-match / strain table)
- ``io_report.txt`` (global report + per-match sections)
- ``stereographic.jpg``, ``stereographic_interactive.html``
- ``pairs_best_it/`` — relaxed best-interface POSCARs (one term per match)
"""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _resolve_artifact_source_root(run_dir: Path) -> Path:
    """
    Artifacts (stereo, pkl, csv) are written to the job directory (cwd during the run).
    If ``run_dir`` already points at ``<job>/result``, real files usually live in ``<job>/``.
    """
    run = run_dir.resolve()
    if not run.name == "result" or not run.is_dir():
        return run
    parent = run.parent
    if (parent / "stereographic.jpg").is_file() or (parent / "stereographic_interactive.html").is_file():
        return parent
    if (
        (parent / "opt_results.pkl").is_file()
        or (parent / "opt_results_.pkl").is_file()
        or (parent / "results.csv").is_file()
        or (parent / "area_strain").is_file()
    ):
        return parent
    if (parent / "io_report.txt").is_file() and not (run / "stereographic.jpg").is_file():
        return parent
    return run


def _append_per_match_sections(run_dir: Path, base_text: str) -> str:
    """Append opt_results-derived sections for each (match, term) in pairs_summary."""
    pkl = run_dir / "opt_results.pkl"
    summary = run_dir / "pairs_summary.txt"
    if not pkl.is_file():
        from InterOptimus.jobflow import resolve_opt_results_pickle_path

        alt = resolve_opt_results_pickle_path(run_dir)
        if alt:
            pkl = Path(alt)
    if not pkl.is_file():
        return base_text
    try:
        from InterOptimus.jobflow import load_opt_results_pickle_payload

        payload = load_opt_results_pickle_payload(str(pkl))
    except Exception:
        return base_text

    opt_results: Dict[Any, Any] = payload.get("opt_results") or {}
    lines: List[str] = [base_text.rstrip(), "", "=" * 80, "Per-match (lowest-energy term) details", "=" * 80, ""]

    rows: List[Tuple[int, int]] = []
    if summary.is_file():
        try:
            txt = summary.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in txt[1:]:
                parts = line.split("\t")
                if len(parts) >= 2:
                    try:
                        rows.append((int(parts[0]), int(parts[1])))
                    except ValueError:
                        continue
        except OSError:
            pass
    if not rows:
        for k in opt_results:
            if isinstance(k, tuple) and len(k) == 2:
                rows.append((int(k[0]), int(k[1])))
        rows.sort()

    for i, j in rows:
        block = opt_results.get((i, j))
        if not isinstance(block, dict):
            continue
        lines.append(f"--- Match {i}, term {j} ---")
        for k in ("relaxed_min_it_E", "relaxed_min_bd_E"):
            if k in block and block[k] is not None:
                lines.append(f"  {k}: {block[k]}")
        rb = block.get("relaxed_best_interface")
        if isinstance(rb, dict) and rb.get("structure") is not None:
            lines.append("  relaxed_best_interface: structure exported to pairs_best_it/")
        lines.append("")
    return "\n".join(lines) + "\n"


def finalize_desktop_result_bundle(run_dir: os.PathLike[str]) -> Path:
    """
    Create ``<job>/result/`` with only the files the desktop GUI exports.

    ``run_dir`` may be either the job directory or an existing ``.../result`` folder; sources
    are resolved to the directory that actually contains run outputs.

    Returns the absolute path to the result bundle directory (``<job>/result``).
    """
    raw = Path(run_dir).resolve()
    src = _resolve_artifact_source_root(raw)
    out = src / "result"
    if out.is_dir():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) results.csv
    src_csv = src / "results.csv"
    legacy = src / "area_strain"
    if src_csv.is_file():
        shutil.copy2(src_csv, out / "results.csv")
    elif legacy.is_file():
        _legacy_area_strain_to_csv(legacy, out / "results.csv")

    # 2) stereographic outputs
    for name in ("stereographic.jpg", "stereographic_interactive.html"):
        p = src / name
        if p.is_file():
            shutil.copy2(p, out / name)

    # 3) io_report.txt — full text + per-match
    rep_parts: List[str] = []
    for candidate in (src / "io_report.txt",):
        if candidate.is_file():
            try:
                rep_parts.append(candidate.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    base = "\n\n".join(rep_parts) if rep_parts else ""
    merged = _append_per_match_sections(src, base)
    (out / "io_report.txt").write_text(merged, encoding="utf-8")

    # 4) POSCAR trees: only pairs_best_it (materialized by local_workflow)
    pb = src / "pairs_best_it"
    if pb.is_dir():
        shutil.copytree(pb, out / "pairs_best_it")

    return out


def _legacy_area_strain_to_csv(src: Path, dst: Path) -> None:
    """Convert legacy ``area_strain`` text lines to CSV."""
    rows_out: List[List[Any]] = []
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                fo = line.find("(")
                fc = line.find(")")
                so = line.find("(", fc + 1)
                sc = line.find(")", so + 1)
                p1 = [float(x) for x in line[fo + 1 : fc].split()]
                p2 = [float(x) for x in line[so + 1 : sc].split()]
                rest = line[sc + 1 :].split()
                if len(rest) < 4:
                    continue
                area = float(rest[0])
                strain = float(rest[1])
                energy = float(rest[2])
                mid = int(rest[3])
                tid = int(rest[4]) if len(rest) > 4 else ""
                rows_out.append(p1 + p2 + [area, strain, energy, mid, tid])
            except (ValueError, IndexError):
                continue
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "film_h",
                "film_k",
                "film_l",
                "substrate_h",
                "substrate_k",
                "substrate_l",
                "area",
                "von_mises_strain",
                "energy_eV",
                "match_id",
                "term_id",
            ]
        )
        for row in rows_out:
            w.writerow(row)
