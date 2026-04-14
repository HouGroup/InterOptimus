"""
Subprocess entry for MatRIS workflow: allows the GUI to terminate the child process on cancel.

Run as: python -m InterOptimus.desktop_app.worker <config.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from InterOptimus.web.local_workflow import run_matris_session


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "missing config path"}))
        sys.exit(2)
    path = Path(sys.argv[1])
    cfg = json.loads(path.read_text(encoding="utf-8"))
    film = Path(cfg["film"])
    sub = Path(cfg["sub"])
    form = cfg["form"]
    out = run_matris_session(film_cif_path=film, substrate_cif_path=sub, form=form)
    print(json.dumps(out, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
