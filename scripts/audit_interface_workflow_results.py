#!/usr/bin/env python3
"""Audit structural invariants and VASP products from remote validation jobs."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure

from InterOptimus.agents.remote_submit import query_interoptimus_task_progress
from InterOptimus.jobflow import (
    load_opt_results_pickle_payload,
    resolve_opt_results_pickle_path,
)


def _structure(value: Any) -> Structure:
    if isinstance(value, Structure):
        return value
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        return Structure.from_dict(value)
    raise TypeError(f"Unsupported stored structure type: {type(value).__name__}")


def _result_file(run_dir: Path | None, name: str) -> Path | None:
    if run_dir is None:
        return None
    plain = run_dir / name
    compressed = run_dir / f"{name}.gz"
    if plain.is_file():
        return plain
    if compressed.is_file():
        return compressed
    return None


def _read_result_bytes(path: Path | None) -> bytes:
    if path is None:
        return b""
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def audit_mlip(name: str, uuid: str) -> dict[str, Any]:
    progress = query_interoptimus_task_progress(uuid)
    run_dir = progress.get("run_dir")
    pickle_path = resolve_opt_results_pickle_path(run_dir) if run_dir else None
    if not pickle_path:
        return {
            "name": name,
            "uuid": uuid,
            "state": progress.get("job_state"),
            "error": "opt_results pickle not found",
        }

    payload = load_opt_results_pickle_payload(pickle_path)
    results = payload["opt_results"]
    pair_audits = []
    failures = []
    for key, result in results.items():
        if not isinstance(result, dict):
            continue
        if result.get("failure"):
            failures.append({"pair": str(key), **result["failure"]})
        best = (result.get("relaxed_best_interface") or {}).get("structure")
        if best is None:
            continue
        structure = _structure(best)
        rb = result.get("relaxed_best_interface") or {}
        film_indices = list(rb.get("film_indices") or result.get("film_indices") or [])
        substrate_indices = list(
            rb.get("substrate_indices") or result.get("substrate_indices") or []
        )
        film_set = set(map(int, film_indices))
        substrate_set = set(map(int, substrate_indices))
        distances = np.asarray(structure.distance_matrix, dtype=float)
        np.fill_diagonal(distances, np.inf)
        energy = rb.get("e")
        pair_audits.append(
            {
                "pair": str(key),
                "atoms": len(structure),
                "film_atoms": len(film_set),
                "substrate_atoms": len(substrate_set),
                "partition_complete": (
                    not film_set.intersection(substrate_set)
                    and film_set.union(substrate_set) == set(range(len(structure)))
                ),
                "finite_lattice": bool(np.isfinite(structure.lattice.matrix).all()),
                "positive_volume": bool(structure.volume > 0),
                "minimum_distance_A": float(np.min(distances)),
                "finite_energy": bool(energy is not None and np.isfinite(float(energy))),
            }
        )

    return {
        "name": name,
        "uuid": uuid,
        "state": progress.get("job_state"),
        "run_dir": run_dir,
        "match_count": len(payload.get("unique_matches_millers") or []),
        "materialized_pair_count": len(payload.get("materialize_pairs") or []),
        "audited_pair_count": len(pair_audits),
        "failures": failures,
        "pairs": pair_audits,
        "all_invariants_pass": bool(pair_audits)
        and all(
            pair["partition_complete"]
            and pair["finite_lattice"]
            and pair["positive_volume"]
            and pair["minimum_distance_A"] > 0.5
            and pair["finite_energy"]
            for pair in pair_audits
        ),
    }


def audit_vasp(name: str, uuid: str) -> dict[str, Any]:
    progress = query_interoptimus_task_progress(uuid)
    jobs = progress.get("expanded_vasp_jobs") or []
    calculation_jobs = [
        job for job in jobs if job.get("name") in {"relax", "static"}
    ]
    product_audits = []
    for job in calculation_jobs:
        run_dir = Path(job["run_dir"]) if job.get("run_dir") else None
        vasprun = _result_file(run_dir, "vasprun.xml")
        outcar = _result_file(run_dir, "OUTCAR")
        incar = _result_file(run_dir, "INCAR")
        vasprun_bytes = _read_result_bytes(vasprun)
        outcar_text = _read_result_bytes(outcar).decode(errors="replace")
        incar_text = _read_result_bytes(incar).decode(errors="replace")
        product_audits.append(
            {
                "uuid": job.get("uuid"),
                "kind": job.get("name"),
                "state": job.get("state"),
                "vasprun_complete": bool(
                    vasprun
                    and len(vasprun_bytes) > 1000
                    and vasprun_bytes.rstrip().endswith(b"</modeling>")
                ),
                "outcar_complete": "General timing and accounting informations"
                in outcar_text,
                "dipole_enabled": "LDIPOL" in incar_text
                and "T" in next(
                    (line for line in incar_text.splitlines() if "LDIPOL" in line),
                    "",
                ),
            }
        )

    counts = progress.get("expanded_vasp_job_counts") or {}
    return {
        "name": name,
        "uuid": uuid,
        "state": progress.get("job_state"),
        "finished": progress.get("is_finished"),
        "job_counts": counts,
        "calculation_job_count": len(calculation_jobs),
        "products": product_audits,
        "all_products_complete": bool(product_audits)
        and all(
            product["state"] == "COMPLETED"
            and product["vasprun_complete"]
            and product["outcar_complete"]
            for product in product_audits
        ),
        "dipole_job_count": sum(
            product["dipole_enabled"] for product in product_audits
        ),
    }


def _parse_named_uuid(value: str) -> tuple[str, str]:
    name, separator, uuid = value.partition("=")
    if not separator or not name or not uuid:
        raise argparse.ArgumentTypeError("expected NAME=UUID")
    return name, uuid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlip", action="append", type=_parse_named_uuid, default=[])
    parser.add_argument("--vasp", action="append", type=_parse_named_uuid, default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "mlip": [audit_mlip(name, uuid) for name, uuid in args.mlip],
        "vasp": [audit_vasp(name, uuid) for name, uuid in args.vasp],
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
