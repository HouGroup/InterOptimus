#!/usr/bin/env bash
# Build InterOptimus.app (macOS) with PyTorch + PyG wheels aligned to the same torch ABI.
#
# Conda: activate your env first, e.g.  conda activate eqnorm
# Optional: export PYTHON=/path/to/python  to force an interpreter.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  PY="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
  # Prefer `python` so conda env wins over Homebrew `python3` (PEP 668 / wrong env).
  PY=python
else
  PY=python3
fi

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: no python on PATH (tried PYTHON, python, python3)." >&2
  exit 1
fi

echo "==> using interpreter: $($PY -c 'import sys; print(sys.executable)')"

echo "==> editable install + pyinstaller"
"$PY" -m pip install -e ".[desktop]"

echo "==> install PyG wheels matching current torch (fixes torch_scatter ABI crashes)"
"$PY" desktop/prep_pyg_stack.py --install

echo "==> verify imports"
"$PY" desktop/verify_torch_stack.py

echo "==> PyInstaller"
"$PY" -m PyInstaller desktop/interoptimus_desktop.spec

echo "OK: dist/InterOptimus.app (and dist/interoptimus_desktop/)"
