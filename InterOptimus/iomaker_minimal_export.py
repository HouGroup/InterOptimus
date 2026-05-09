"""
Minimal MLIP / VASP export folders: each contains only

- ``all_match_info`` — stereographic table (same format as legacy ``area_strain``);
- ``stereographic_interactive`` — interactive HTML (no ``.html`` suffix);
- ``stereographic.jpg``;
- ``selected_interfaces.csv`` — same columns as ``InterfaceWorker.global_minimization`` output
  ``all_data_*.csv`` (i.e. ``global_optimized_data.to_csv``), plus a final ``interface_cif`` column
  from ``opt_results[(i_m, i_t)]['relaxed_best_interface']['structure']``.

Heavy artifacts (``opt_results.pkl``, ``pairs_best_it/``, reports) stay at the **export root**
(``*_results/`` or ``fetched_results/``), not inside these two subfolders.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FN_ALL_MATCH_INFO = "all_match_info"
FN_STEREO_INTERACTIVE = "stereographic_interactive"
FN_STEREO_JPG = "stereographic.jpg"
FN_SELECTED_CSV = "selected_interfaces.csv"

# Match :class:`InterfaceWorker.global_minimization` DataFrame column names (``all_data_*.csv``).
_G_COL_IM = r"$i_m$"
_G_COL_IT = r"$i_t$"


def _opt_results_pair_key(opt_results: Dict[Any, Any], mi: int, ti: int) -> Optional[Any]:
    for k in ((mi, ti), (str(mi), str(ti))):
        if k in opt_results:
            return k
    return None


def _interface_cif_for_opt_pair(opt_results: Dict[Any, Any], mi: int, ti: int) -> str:
    pk = _opt_results_pair_key(opt_results or {}, mi, ti)
    if pk is None:
        return ""
    ent = (opt_results or {}).get(pk)
    if not isinstance(ent, dict):
        return ""
    rb = ent.get("relaxed_best_interface")
    if not isinstance(rb, dict):
        return ""
    st = _ensure_structure(rb.get("structure"))
    return _structure_to_cif_text(st)


def _find_newest_all_data_csv(*roots: str) -> Optional[str]:
    """Newest ``all_data*.csv`` under any *roots* (same basename as ``global_optimized_data.to_csv``)."""
    from pathlib import Path

    candidates: List[Path] = []
    for root in roots:
        if not root:
            continue
        rp = Path(root)
        if not rp.is_dir():
            continue
        try:
            for p in rp.rglob("all_data*.csv"):
                if p.is_file():
                    candidates.append(p)
        except OSError:
            continue
    if not candidates:
        return None
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(best.resolve())


def write_mlip_selected_interfaces_csv_table(df: Any, opt_results: Dict[Any, Any], path: str) -> None:
    """
    Write ``selected_interfaces.csv``: identical to ``global_optimized_data.to_csv`` layout,
    plus final column ``interface_cif`` from MLIP-relaxed structures in ``opt_results``.
    """
    import pandas as pd

    if df is None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("", encoding="utf-8")
        return
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    try:
        n = len(df)
    except TypeError:
        n = 0
    if n == 0:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("", encoding="utf-8")
        return
    out = df.copy()
    if "interface_cif" in out.columns:
        out = out.drop(columns=["interface_cif"])
    im_c, it_c = _G_COL_IM, _G_COL_IT
    if im_c not in out.columns or it_c not in out.columns:
        out["interface_cif"] = ""
    else:
        cifs: List[str] = []
        for _, row in out.iterrows():
            try:
                mi = int(float(row[im_c]))
                ti = int(float(row[it_c]))
            except (TypeError, ValueError):
                cifs.append("")
                continue
            cifs.append(_interface_cif_for_opt_pair(opt_results or {}, mi, ti))
        out["interface_cif"] = cifs
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")


def _structure_to_cif_text(st: Any) -> str:
    """Serialize a :class:`pymatgen.core.structure.Structure` to a CIF string; empty on failure."""
    if st is None:
        return ""
    try:
        out = st.to(fmt="cif")
    except Exception:
        return ""
    if isinstance(out, bytes):
        return out.decode("utf-8", errors="replace")
    return str(out) if out is not None else ""


def purge_extraneous_from_results_subdir(dir_path: str) -> None:
    """
    Remove everything under *dir_path* except the minimal export set:

    ``all_match_info``, ``stereographic_interactive``, ``stereographic.jpg``,
    ``selected_interfaces.csv``.
    """
    p = Path(dir_path)
    if not p.is_dir():
        return
    allowed = {
        FN_ALL_MATCH_INFO,
        FN_STEREO_INTERACTIVE,
        FN_STEREO_JPG,
        FN_SELECTED_CSV,
    }
    for child in list(p.iterdir()):
        name = child.name
        if name in allowed:
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except OSError:
            pass


def _uvw12_from_miller_dict(miller: Dict[str, Any]) -> Tuple[str, ...]:
    """Twelve vector components: film v1,v2 then substrate v1,v2 (each h,k,l as str)."""
    empty = ("",) * 12
    if not isinstance(miller, dict):
        return empty
    try:
        fv = miller.get("film_conventional_vectors") or []
        sv = miller.get("substrate_conventional_vectors") or []
        if len(fv) >= 2 and len(sv) >= 2:
            parts: List[str] = []
            for row in (fv[0], fv[1], sv[0], sv[1]):
                if len(row) < 3:
                    return empty
                for j in range(3):
                    v = row[j]
                    parts.append(str(int(v)) if float(v) == int(float(v)) else str(float(v)))
            if len(parts) == 12:
                return tuple(parts)  # type: ignore[return-value]
    except (TypeError, ValueError, IndexError):
        pass
    return empty


def _ensure_structure(obj: Any):
    if obj is None:
        return None
    try:
        from pymatgen.core.structure import Structure
    except ImportError:
        return None
    if isinstance(obj, Structure):
        return obj
    if isinstance(obj, dict):
        try:
            return Structure.from_dict(obj)
        except Exception:
            return None
    if isinstance(obj, str):
        try:
            return Structure.from_dict(__import__("json").loads(obj))
        except Exception:
            return None
    return None


def _load_opt_results_summary_by_pair(export_root: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Parse ``opt_results_summary.json`` (export root or ``mlip_results/``) into ``(match, term) → row``."""
    root = os.path.abspath(export_root)
    for rel in ("opt_results_summary.json", os.path.join("mlip_results", "opt_results_summary.json")):
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        out: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for v in raw.values():
            if not isinstance(v, dict):
                continue
            try:
                mi, ti = int(v["match_id"]), int(v["term_id"])
            except (KeyError, TypeError, ValueError):
                continue
            out[(mi, ti)] = v
        if out:
            return out
    return {}


def _merge_unique_matches_millers_from_sidecar(
    payload_um: Any,
    export_root: str,
) -> List[Dict[str, Any]]:
    """
    Fill missing Miller / UVW columns using ``unique_matches_millers.json`` next to the pickle.

    Sidecar is merged entry-wise; pickle values win on key clashes, but vectors are copied
    from the sidecar when absent in the merged row.
    """
    p_rows: List[Dict[str, Any]] = [x for x in payload_um if isinstance(x, dict)] if isinstance(payload_um, list) else []
    root = os.path.abspath(export_root)
    side: List[Dict[str, Any]] = []
    for rel in ("unique_matches_millers.json", os.path.join("mlip_results", "unique_matches_millers.json")):
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            side = [x for x in data if isinstance(x, dict)]
            break
    if not side:
        return p_rows
    n = max(len(p_rows), len(side))
    out: List[Dict[str, Any]] = []
    for i in range(n):
        s = side[i] if i < len(side) else {}
        pr = p_rows[i] if i < len(p_rows) else {}
        merged = dict(s)
        merged.update(pr)
        if not merged.get("film_conventional_vectors") and s.get("film_conventional_vectors"):
            merged["film_conventional_vectors"] = s["film_conventional_vectors"]
        if not merged.get("substrate_conventional_vectors") and s.get("substrate_conventional_vectors"):
            merged["substrate_conventional_vectors"] = s["substrate_conventional_vectors"]
        if not merged.get("film_conventional_miller") and s.get("film_conventional_miller"):
            merged["film_conventional_miller"] = s["film_conventional_miller"]
        if not merged.get("substrate_conventional_miller") and s.get("substrate_conventional_miller"):
            merged["substrate_conventional_miller"] = s["substrate_conventional_miller"]
        out.append(merged)
    return out


def _csv_float(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(float(x))
    except (TypeError, ValueError):
        return str(x)


def build_selected_interfaces_rows_mlip(
    payload: Dict[str, Any],
    *,
    interface_cif_text_by_pair: Dict[Tuple[int, int], str],
    summary_lookup: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """One row per relaxed interface in ``materialize_pairs`` (aligned with ``global_minimization`` table)."""
    opt_results: Dict[Any, Any] = payload.get("opt_results") or {}
    um: List[Dict[str, Any]] = payload.get("unique_matches_millers") or []
    if not isinstance(um, list):
        um = []
    keys: List[Tuple[int, int]] = []
    raw_pairs = payload.get("materialize_pairs") or []
    for item in raw_pairs:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            keys.append((int(item[0]), int(item[1])))
    if not keys:
        for k in opt_results:
            if isinstance(k, tuple) and len(k) == 2:
                try:
                    kk = (int(k[0]), int(k[1]))
                except (TypeError, ValueError):
                    continue
                rb = (opt_results.get(k) or {}).get("relaxed_best_interface") or {}
                if rb.get("structure") is not None:
                    keys.append(kk)
    rows: List[Dict[str, str]] = []
    for mi, ti in sorted(set(keys)):
        od = opt_results.get((mi, ti))
        if od is None:
            od = opt_results.get((str(mi), str(ti))) or {}
        if not isinstance(od, dict):
            continue
        rb = od.get("relaxed_best_interface") or {}
        if rb.get("structure") is None:
            continue
        miller = um[mi] if 0 <= mi < len(um) and isinstance(um[mi], dict) else {}
        fh, fk, fl = "", "", ""
        sh, sk, sl = "", "", ""
        try:
            fc = miller.get("film_conventional_miller") or []
            sc = miller.get("substrate_conventional_miller") or []
            if len(fc) >= 3:
                fh, fk, fl = str(int(fc[0])), str(int(fc[1])), str(int(fc[2]))
            if len(sc) >= 3:
                sh, sk, sl = str(int(sc[0])), str(int(sc[1])), str(int(sc[2]))
        except (TypeError, ValueError, IndexError):
            pass
        u_f1, v_f1, w_f1, u_f2, v_f2, w_f2, u_s1, v_s1, w_s1, u_s2, v_s2, w_s2 = _uvw12_from_miller_dict(miller)
        struct = _ensure_structure(rb.get("structure"))
        area = ""
        if struct is not None:
            try:
                area = _csv_float(struct.lattice.a * struct.lattice.b)
            except Exception:
                area = ""
        strain = _csv_float(od.get("strain"))
        double_it = bool(payload.get("double_interface"))
        ev = od.get("relaxed_min_it_E") if double_it else od.get("relaxed_min_bd_E")
        if ev is None:
            rb0 = od.get("relaxed_best_interface") if isinstance(od.get("relaxed_best_interface"), dict) else {}
            if isinstance(rb0, dict) and rb0.get("e") is not None:
                try:
                    ev = float(rb0["e"])
                except (TypeError, ValueError):
                    ev = None
        strain_e = _csv_float(od.get("strain_E")) if od.get("strain_E") is not None else ""
        disp = rb.get("mlip_displacement") or {}
        film_d = _csv_float(disp.get("film_avg_disp")) if disp.get("film_avg_disp") is not None else ""
        sub_d = _csv_float(disp.get("substrate_avg_disp")) if disp.get("substrate_avg_disp") is not None else ""
        cif_body = (interface_cif_text_by_pair or {}).get((mi, ti), "")

        sumd = (summary_lookup or {}).get((mi, ti))
        if sumd:
            if not area:
                ma = sumd.get("match_area")
                if ma is not None:
                    area = _csv_float(ma)
            if not strain:
                st = sumd.get("strain")
                if st is not None:
                    strain = _csv_float(st)
            if ev is None:
                ev = sumd.get("relaxed_min_it_E") if double_it else sumd.get("relaxed_min_bd_E")
                if ev is None:
                    ev = sumd.get("relaxed_best_interface_E")
                if ev is not None:
                    try:
                        ev = float(ev)
                    except (TypeError, ValueError):
                        ev = None
            if not strain_e and sumd.get("strain_E") is not None:
                strain_e = _csv_float(sumd.get("strain_E"))
            if (not film_d or not sub_d) and isinstance(sumd.get("mlip_displacement"), dict):
                sd = sumd["mlip_displacement"]
                if not film_d and sd.get("film_avg_disp") is not None:
                    film_d = _csv_float(sd.get("film_avg_disp"))
                if not sub_d and sd.get("substrate_avg_disp") is not None:
                    sub_d = _csv_float(sd.get("substrate_avg_disp"))
        rows.append(
            {
                "match_id": str(mi),
                "term_id": str(ti),
                "film_h": fh,
                "film_k": fk,
                "film_l": fl,
                "substrate_h": sh,
                "substrate_k": sk,
                "substrate_l": sl,
                "match_area_A2": area,
                "von_mises_strain": strain,
                "energy_J_m2": _csv_float(ev) if ev is not None else "",
                "strain_E_eV_per_atom": strain_e,
                "u_f1": u_f1,
                "v_f1": v_f1,
                "w_f1": w_f1,
                "u_f2": u_f2,
                "v_f2": v_f2,
                "w_f2": w_f2,
                "u_s1": u_s1,
                "v_s1": v_s1,
                "w_s1": w_s1,
                "u_s2": u_s2,
                "v_s2": v_s2,
                "w_s2": w_s2,
                "termination": "",
                "film_avg_disp_mlip_A": film_d,
                "substrate_avg_disp_mlip_A": sub_d,
                "interface_cif": cif_body,
            }
        )
    return rows


def build_selected_interfaces_rows_vasp(
    pair_summary_rows: List[Dict[str, Any]],
    *,
    interface_cif_text_by_pair: Dict[Tuple[int, int], str],
    pkl_payload: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    um: List[Dict[str, Any]] = []
    if isinstance(pkl_payload, dict):
        raw = pkl_payload.get("unique_matches_millers")
        if isinstance(raw, list):
            um = raw
    rows: List[Dict[str, str]] = []
    for row in pair_summary_rows:
        mi = int(row["match_id"])
        ti = int(row["term_id"])
        cif_body = (interface_cif_text_by_pair or {}).get((mi, ti), "")
        fm = row.get("film_conventional_miller") or []
        sm = row.get("substrate_conventional_miller") or []
        fh, fk, fl = ("", "", "")
        sh, sk, sl = ("", "", "")
        try:
            if len(fm) >= 3:
                fh, fk, fl = str(int(fm[0])), str(int(fm[1])), str(int(fm[2]))
            if len(sm) >= 3:
                sh, sk, sl = str(int(sm[0])), str(int(sm[1])), str(int(sm[2]))
        except (TypeError, ValueError):
            pass
        miller = um[mi] if 0 <= mi < len(um) and isinstance(um[mi], dict) else {}
        u_f1, v_f1, w_f1, u_f2, v_f2, w_f2, u_s1, v_s1, w_s1, u_s2, v_s2, w_s2 = _uvw12_from_miller_dict(miller)
        ev = row.get("energy_value")
        evs = "" if ev is None or str(ev) == "N/A" else _csv_float(ev)
        ek = str(row.get("energy_type") or "interface_energy")
        rows.append(
            {
                "match_id": str(mi),
                "term_id": str(ti),
                "film_h": fh,
                "film_k": fk,
                "film_l": fl,
                "substrate_h": sh,
                "substrate_k": sk,
                "substrate_l": sl,
                "u_f1": u_f1,
                "v_f1": v_f1,
                "w_f1": w_f1,
                "u_f2": u_f2,
                "v_f2": v_f2,
                "w_f2": w_f2,
                "u_s1": u_s1,
                "v_s1": v_s1,
                "w_s1": w_s1,
                "u_s2": u_s2,
                "v_s2": v_s2,
                "w_s2": w_s2,
                "match_area_A2": _csv_float(row.get("match_area")),
                "von_mises_strain": _csv_float(row.get("strain")),
                "energy_kind": ek,
                "energy_J_m2": evs,
                "strain_E_eV_per_atom": "",
                "film_avg_disp_mlip_A": _csv_float(row.get("film_avg_disp_mlip_A")),
                "substrate_avg_disp_mlip_A": _csv_float(row.get("substrate_avg_disp_mlip_A")),
                "film_avg_disp_vasp_A": _csv_float(row.get("film_avg_disp_vasp_A")),
                "substrate_avg_disp_vasp_A": _csv_float(row.get("substrate_avg_disp_vasp_A")),
                "vasp_dft_status": str(row.get("vasp_dft_status") or ""),
                "phase": "vasp",
                "interface_cif": cif_body,
            }
        )
    return rows


def write_selected_interfaces_csv(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def interface_cif_text_by_pair_from_payload(payload: Dict[str, Any]) -> Dict[Tuple[int, int], str]:
    """Build ``(match_id, term_id) → CIF string`` from ``opt_results`` relaxed structures (no files)."""
    opt_results: Dict[Any, Any] = payload.get("opt_results") or {}
    keys: List[Tuple[int, int]] = []
    for item in payload.get("materialize_pairs") or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            keys.append((int(item[0]), int(item[1])))
    if not keys:
        for k in opt_results:
            if isinstance(k, tuple) and len(k) == 2:
                try:
                    keys.append((int(k[0]), int(k[1])))
                except (TypeError, ValueError):
                    continue
    out: Dict[Tuple[int, int], str] = {}
    for mi, ti in sorted(set(keys)):
        od = opt_results.get((mi, ti)) or opt_results.get((str(mi), str(ti))) or {}
        if not isinstance(od, dict):
            continue
        st = _ensure_structure((od.get("relaxed_best_interface") or {}).get("structure"))
        if st is None:
            continue
        txt = _structure_to_cif_text(st)
        if txt:
            out[(mi, ti)] = txt
    return out


def ensure_mlip_csv_from_summary_fallback(session_dir: str) -> bool:
    """
    If ``mlip_results/selected_interfaces.csv`` is missing or tiny, try to rebuild it from
    ``opt_results.pkl`` using ``global_optimized_records`` + ``opt_results`` (same as the run node).

    Returns True if a file was written.
    """
    from InterOptimus.jobflow import load_opt_results_pickle_payload

    wr = Path(os.path.abspath(session_dir))
    if not wr.is_dir():
        return False
    todo: List[Tuple[str, str]] = []
    fr = wr / "fetched_results"
    if fr.is_dir():
        mlip_sub = fr / "mlip_results"
        exp = str(fr.resolve())
        mlip_dir = str(mlip_sub.resolve())
        todo.append((exp, mlip_dir))
    top_mlip = wr / "mlip_results"
    if top_mlip.is_dir():
        todo.append((str(wr.resolve()), str(top_mlip.resolve())))
    written = False
    for export_root, mlip_dir in todo:
        csv_p = Path(mlip_dir) / FN_SELECTED_CSV
        if csv_p.is_file() and csv_p.stat().st_size > 32:
            continue
        pkl = os.path.join(export_root, "opt_results.pkl")
        if not os.path.isfile(pkl):
            pkl = os.path.join(mlip_dir, "opt_results.pkl")
        if not os.path.isfile(pkl):
            continue
        try:
            payload = load_opt_results_pickle_payload(pkl)
        except Exception:
            continue
        gor = payload.get("global_optimized_records")
        if not isinstance(gor, list):
            gor = []
        import pandas as pd

        df_out = None
        ad = _find_newest_all_data_csv(export_root)
        if ad:
            try:
                df_out = pd.read_csv(ad, encoding="utf-8")
            except Exception:
                df_out = None
        if (df_out is None or len(df_out) == 0) and gor:
            try:
                df_out = pd.DataFrame(gor)
            except Exception:
                df_out = None
        if df_out is None or len(df_out) == 0:
            continue
        os.makedirs(mlip_dir, exist_ok=True)
        write_mlip_selected_interfaces_csv_table(df_out, payload.get("opt_results") or {}, str(csv_p))
        written = True
    return written


def write_mlip_minimal_results_folder(export_root: str, *, run_dir: str) -> Dict[str, Any]:
    """
    Populate ``<export_root>/mlip_results/`` with the four public files (including CSV with embedded CIFs).

    Reads stereographic inputs from *run_dir* and ``opt_results.pkl`` under *export_root*.
    """
    from InterOptimus.jobflow import load_opt_results_pickle_payload

    export_root = os.path.abspath(export_root)
    mlip_dir = os.path.join(export_root, "mlip_results")
    os.makedirs(mlip_dir, exist_ok=True)
    purge_extraneous_from_results_subdir(mlip_dir)
    rd = os.path.abspath(run_dir)
    out: Dict[str, Any] = {"mlip_results_dir": mlip_dir, "files": []}

    area_src = os.path.join(rd, "area_strain")
    if os.path.isfile(area_src):
        shutil.copy2(area_src, os.path.join(mlip_dir, FN_ALL_MATCH_INFO))
        out["files"].append(FN_ALL_MATCH_INFO)

    for img in (FN_STEREO_JPG,):
        src = os.path.join(rd, img)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(mlip_dir, img))
            out["files"].append(img)

    html_src = os.path.join(rd, "stereographic_interactive.html")
    dst_int = os.path.join(mlip_dir, FN_STEREO_INTERACTIVE)
    if os.path.isfile(html_src):
        shutil.copy2(html_src, dst_int)
        out["files"].append(FN_STEREO_INTERACTIVE)

    pkl = os.path.join(export_root, "opt_results.pkl")
    if not os.path.isfile(pkl):
        pkl = os.path.join(export_root, "mlip_results", "opt_results.pkl")
    if os.path.isfile(pkl):
        try:
            import pandas as pd

            payload = load_opt_results_pickle_payload(pkl)
            opt_res = payload.get("opt_results") or {}
            gor = payload.get("global_optimized_records")
            if not isinstance(gor, list):
                gor = []
            df_tab = None
            ad = _find_newest_all_data_csv(export_root, rd)
            if ad:
                try:
                    df_tab = pd.read_csv(ad, encoding="utf-8")
                except Exception:
                    df_tab = None
            if (df_tab is None or len(df_tab) == 0) and gor:
                try:
                    df_tab = pd.DataFrame(gor)
                except Exception:
                    df_tab = None
            csv_path = os.path.join(mlip_dir, FN_SELECTED_CSV)
            if df_tab is not None and len(df_tab) > 0:
                write_mlip_selected_interfaces_csv_table(df_tab, opt_res, csv_path)
                out["files"].append(FN_SELECTED_CSV)
                if _G_COL_IM in df_tab.columns and _G_COL_IT in df_tab.columns:
                    nc = 0
                    for _, row in df_tab.iterrows():
                        try:
                            mi = int(float(row[_G_COL_IM]))
                            ti = int(float(row[_G_COL_IT]))
                        except (TypeError, ValueError):
                            continue
                        if _interface_cif_for_opt_pair(opt_res, mi, ti).strip():
                            nc += 1
                    out["n_rows_with_interface_cif"] = nc
                else:
                    out["n_rows_with_interface_cif"] = 0
            else:
                summary_lookup = _load_opt_results_summary_by_pair(export_root)
                merged_um = _merge_unique_matches_millers_from_sidecar(
                    payload.get("unique_matches_millers"),
                    export_root,
                )
                payload_eff = dict(payload)
                payload_eff["unique_matches_millers"] = merged_um
                cif_text_by_pair = interface_cif_text_by_pair_from_payload(payload_eff)
                rows = build_selected_interfaces_rows_mlip(
                    payload_eff,
                    interface_cif_text_by_pair=cif_text_by_pair,
                    summary_lookup=summary_lookup or None,
                )
                write_selected_interfaces_csv(csv_path, rows)
                out["files"].append(FN_SELECTED_CSV)
                out["n_rows_with_interface_cif"] = sum(
                    1 for r in rows if (r.get("interface_cif") or "").strip()
                )
        except Exception as e:
            out["selected_csv_error"] = str(e)
    else:
        out["selected_csv_error"] = "missing opt_results.pkl under export root"

    return out


def write_vasp_minimal_results_folder(
    vasp_root: str,
    *,
    all_match_info_text: str,
    stereographic_jpg_src: Optional[str],
    stereographic_interactive_html_src: Optional[str],
    selected_rows: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Write only the minimal set into ``vasp_root`` (overwrite)."""
    vasp_root = os.path.abspath(vasp_root)
    os.makedirs(vasp_root, exist_ok=True)
    for pat in (FN_ALL_MATCH_INFO, FN_STEREO_INTERACTIVE, FN_STEREO_JPG, FN_SELECTED_CSV):
        p = os.path.join(vasp_root, pat)
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass
    for p in Path(vasp_root).glob("match_*_term_*.cif"):
        try:
            p.unlink()
        except OSError:
            pass

    Path(os.path.join(vasp_root, FN_ALL_MATCH_INFO)).write_text(all_match_info_text, encoding="utf-8")
    if stereographic_jpg_src and os.path.isfile(stereographic_jpg_src):
        shutil.copy2(stereographic_jpg_src, os.path.join(vasp_root, FN_STEREO_JPG))
    if stereographic_interactive_html_src and os.path.isfile(stereographic_interactive_html_src):
        shutil.copy2(stereographic_interactive_html_src, os.path.join(vasp_root, FN_STEREO_INTERACTIVE))
    write_selected_interfaces_csv(os.path.join(vasp_root, FN_SELECTED_CSV), selected_rows)
    return {"vasp_results_dir": vasp_root, "files": [FN_ALL_MATCH_INFO, FN_STEREO_INTERACTIVE, FN_STEREO_JPG, FN_SELECTED_CSV]}
