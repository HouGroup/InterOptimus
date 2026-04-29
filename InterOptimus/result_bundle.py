"""
After a successful local run / fetch, populate curated result folders:

- ``mlip_results/``: MLIP-selected interface CSV, ``area_match``, stereographic JPG/HTML.
- ``vasp_results/``: VASP-valued CSV / ``area_match`` / stereographic JPG/HTML (built elsewhere).
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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


def _jsonish(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _scalar_or_json(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return _jsonish(value.tolist())
    except Exception:
        pass
    if isinstance(value, (list, tuple, dict)):
        return _jsonish(value)
    return str(value)


def _structure_cif_from_obj(obj: Any) -> str:
    if obj is None:
        return ""
    try:
        from pymatgen.core.structure import Structure

        if isinstance(obj, Structure):
            return obj.to(fmt="cif")
        if isinstance(obj, dict):
            return Structure.from_dict(obj).to(fmt="cif")
        if isinstance(obj, str):
            try:
                return Structure.from_dict(json.loads(obj)).to(fmt="cif")
            except Exception:
                return obj if obj.lstrip().startswith("data_") else ""
    except Exception:
        return ""
    return ""


def _selected_pair_keys(payload: Dict[str, Any], opt_results: Dict[Any, Any]) -> List[Tuple[int, int]]:
    raw = payload.get("materialize_pairs")
    out: List[Tuple[int, int]] = []
    if isinstance(raw, list):
        for item in raw:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    out.append((int(item[0]), int(item[1])))
            except (TypeError, ValueError):
                continue
    if out:
        return out
    for key in opt_results:
        if isinstance(key, tuple) and len(key) == 2:
            try:
                out.append((int(key[0]), int(key[1])))
            except (TypeError, ValueError):
                continue
    return sorted(out)


def _global_rows_by_pair(payload: Dict[str, Any]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    rows = payload.get("global_optimized_data")
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            pair = (int(row.get("$i_m$")), int(row.get("$i_t$")))
        except (TypeError, ValueError):
            continue
        clean: Dict[str, Any] = {}
        for key, value in row.items():
            clean[str(key)] = _scalar_or_json(value)
        out[pair] = clean
    return out


def _global_rows_by_pair_from_all_data(src: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Fallback for old runs whose ``opt_results.pkl`` lacks ``global_optimized_data``."""
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for path in sorted(src.glob("all_data*.csv")):
        if not path.is_file():
            continue
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for raw in reader:
                    row = {
                        str(k): _scalar_or_json(v)
                        for k, v in raw.items()
                        if k is not None and str(k) and not str(k).startswith("Unnamed:")
                    }
                    try:
                        pair = (int(float(row.get("$i_m$"))), int(float(row.get("$i_t$"))))
                    except (TypeError, ValueError):
                        continue
                    out[pair] = row
        except OSError:
            continue
        if out:
            return out
    return out


def _row_from_opt_result(
    match_id: int,
    term_id: int,
    entry: Dict[str, Any],
    global_row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = dict(global_row) if global_row else {"$i_m$": match_id, "$i_t$": term_id}
    if global_row:
        row["$i_m$"] = row.get("$i_m$", match_id)
        row["$i_t$"] = row.get("$i_t$", term_id)
    rb = entry.get("relaxed_best_interface")
    if isinstance(rb, dict):
        row["cif"] = _structure_cif_from_obj(rb.get("structure"))
    else:
        row["cif"] = ""
    return row


def _write_rows_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows_list = list(rows)
    rows_list.sort(key=_interface_energy_sort_key)
    cols: List[str] = []
    seen = set()
    for row in rows_list:
        for key in row:
            if key not in seen:
                seen.add(key)
                cols.append(key)
    if "cif" in cols:
        cols = [c for c in cols if c != "cif"] + ["cif"]
    elif not cols:
        cols = ["cif"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def _interface_energy_sort_key(row: Dict[str, Any]) -> Tuple[float, int, int]:
    """Sort selected interfaces by interface/cohesive energy, missing values last."""
    for key, value in row.items():
        if "E_{it}" in str(key) or "E_{bd}" in str(key):
            try:
                return (float(value), int(row.get("$i_m$") or 0), int(row.get("$i_t$") or 0))
            except (TypeError, ValueError):
                continue
    for key in (
        "relaxed_min_it_E",
        "relaxed_min_bd_E",
        "energy_value",
        "mlip_energy",
        "vasp_energy",
    ):
        value = row.get(key)
        try:
            return (float(value), int(row.get("$i_m$") or row.get("match_id") or 0), int(row.get("$i_t$") or row.get("term_id") or 0))
        except (TypeError, ValueError):
            continue
    return (float("inf"), int(row.get("$i_m$") or row.get("match_id") or 0), int(row.get("$i_t$") or row.get("term_id") or 0))


def write_mlip_results_bundle(run_dir: os.PathLike[str]) -> Path:
    """Create ``mlip_results/`` with selected-interface CSV and MLIP plot artifacts."""
    src = _resolve_artifact_source_root(Path(run_dir))
    out = src / "mlip_results"
    if out.is_dir():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    pkl = src / "opt_results.pkl"
    if not pkl.is_file():
        try:
            from InterOptimus.jobflow import resolve_opt_results_pickle_path

            alt = resolve_opt_results_pickle_path(src)
            if alt:
                pkl = Path(alt)
        except Exception:
            pass

    rows: List[Dict[str, Any]] = []
    if pkl.is_file():
        try:
            from InterOptimus.jobflow import load_opt_results_pickle_payload

            payload = load_opt_results_pickle_payload(str(pkl))
            opt_results: Dict[Any, Any] = payload.get("opt_results") or {}
            global_rows = _global_rows_by_pair(payload)
            if not global_rows:
                global_rows = _global_rows_by_pair_from_all_data(src)
            for match_id, term_id in _selected_pair_keys(payload, opt_results):
                entry = opt_results.get((match_id, term_id))
                if isinstance(entry, dict):
                    rows.append(_row_from_opt_result(match_id, term_id, entry, global_rows.get((match_id, term_id))))
        except Exception as exc:
            rows.append({"error": str(exc), "cif": ""})

    _write_rows_csv(out / "selected_interfaces.csv", rows)

    area_src = src / "area_strain"
    if area_src.is_file():
        shutil.copy2(area_src, out / "area_match")
    for name in ("stereographic.jpg", "stereographic_interactive.html"):
        p = src / name
        if p.is_file():
            shutil.copy2(p, out / name)
    return out


def finalize_session_result_bundle(run_dir: os.PathLike[str]) -> Path:
    """
    Create ``<job>/result/`` with a curated artifact set (CSV, stereographic, report, POSCARs).

    ``run_dir`` may be either the job directory or an existing ``.../result`` folder; sources
    are resolved to the directory that actually contains run outputs.

    Returns the absolute path to the result bundle directory (``<job>/result``).
    """
    raw = Path(run_dir).resolve()
    src = _resolve_artifact_source_root(raw)
    write_mlip_results_bundle(src)
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

    # 4) POSCAR trees: only pairs_best_it (materialized by session_workflow)
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
