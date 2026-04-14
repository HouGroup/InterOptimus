#!/usr/bin/env bash
# Build InterOptimus Desktop with PyInstaller (macOS produces InterOptimus.app).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="$ROOT/desktop/bundled_home/.cache/matris/MatRIS_10M_OAM.pth.tar"

cd "$ROOT"
if [[ ! -f "$CKPT" ]]; then
  echo "Error: missing MatRIS checkpoint: $CKPT" >&2
  echo "Place MatRIS_10M_OAM.pth.tar under desktop/bundled_home/.cache/matris/ before packaging." >&2
  exit 1
fi

exec pyinstaller desktop/interoptimus_desktop.spec
