"""Rebuild ``mlip_results/selected_interfaces.csv`` from ``all_data*.csv`` / pickle + ``opt_results``."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pymatgen.core import Lattice, Structure

from InterOptimus.iomaker_minimal_export import (
    FN_SELECTED_CSV,
    _G_COL_IM,
    ensure_mlip_csv_from_summary_fallback,
    write_mlip_selected_interfaces_csv_table,
)


def _minimal_global_record(it_energy: bool = True) -> dict:
    e_col = r"$E_{it}$ $(J/m^2)$" if it_energy else r"$E_{bd}$ $(J/m^2)$"
    return {
        r"$h_f$": 1,
        r"$k_f$": 0,
        r"$l_f$": 0,
        r"$h_s$": 1,
        r"$k_s$": 0,
        r"$l_s$": 0,
        r"$A$ (" + "\u00C5" + "$^2$)": 12.0,
        r"$\epsilon$": 0.01,
        e_col: 0.42,
        r"$E_{el}$ $(eV/atom)$": 0.0,
        r"$u_{f1}$": 1,
        r"$v_{f1}$": 0,
        r"$w_{f1}$": 0,
        r"$u_{f2}$": 0,
        r"$v_{f2}$": 1,
        r"$w_{f2}$": 0,
        r"$u_{s1}$": 1,
        r"$v_{s1}$": 0,
        r"$w_{s1}$": 0,
        r"$u_{s2}$": 0,
        r"$v_{s2}$": 1,
        r"$w_{s2}$": 0,
        r"$T$": 0,
        r"$i_m$": 0,
        r"$i_t$": 0,
        r"$\bar{d}_{f}^{MLIP}$ (" + "\u00C5" + ")": 0.1,
        r"$\bar{d}_{s}^{MLIP}$ (" + "\u00C5" + ")": 0.2,
    }


class TestSummaryCsvFallback(unittest.TestCase):
    def test_writes_csv_from_pickle_records(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wr = Path(td)
            fr = wr / "fetched_results"
            mlip = fr / "mlip_results"
            mlip.mkdir(parents=True)
            st = Structure(Lattice.cubic(3.0), ["H"], [[0.0, 0.0, 0.0]])
            payload = {
                "version": 1,
                "global_optimized_records": [_minimal_global_record()],
                "opt_results": {
                    (0, 0): {"relaxed_best_interface": {"structure": st.as_dict()}},
                },
            }
            (fr / "opt_results.pkl").write_bytes(pickle.dumps(payload))
            self.assertTrue(ensure_mlip_csv_from_summary_fallback(str(wr)))
            csv_p = mlip / FN_SELECTED_CSV
            self.assertTrue(csv_p.is_file())
            text = csv_p.read_text(encoding="utf-8")
            self.assertIn(_G_COL_IM, text)
            self.assertIn("interface_cif", text)
            self.assertIn("data_", text.lower())

    def test_interface_cif_from_relaxed_structure(self) -> None:
        st = Structure(Lattice.cubic(3.0), ["H"], [[0.0, 0.0, 0.0]])
        df = pd.DataFrame([_minimal_global_record()])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.csv"
            write_mlip_selected_interfaces_csv_table(
                df,
                {(0, 0): {"relaxed_best_interface": {"structure": st.as_dict()}}},
                str(path),
            )
            self.assertTrue(path.is_file())
            body = path.read_text(encoding="utf-8")
            self.assertIn("interface_cif", body)
            self.assertIn("data_", body.lower())

    def test_skips_when_csv_already_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wr = Path(td)
            fr = wr / "fetched_results"
            mlip = fr / "mlip_results"
            mlip.mkdir(parents=True)
            (fr / "opt_results.pkl").write_bytes(
                pickle.dumps({"global_optimized_records": [_minimal_global_record()], "opt_results": {}})
            )
            (mlip / FN_SELECTED_CSV).write_text(
                "match_id,term_id,film_h,film_k,film_l\n0,0,1,0,0\n",
                encoding="utf-8",
            )
            self.assertFalse(ensure_mlip_csv_from_summary_fallback(str(wr)))


if __name__ == "__main__":
    unittest.main()
