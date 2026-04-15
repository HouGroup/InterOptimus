#!/usr/bin/env bash
# 将当前 Python 环境的 PyTorch 固定为 2.6.x，再安装与之一致的 PyG 轮子（Eqnorm / 桌面打包用）。
# 建议在专用 conda 里执行，例如: conda activate eqnorm
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  PY="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi

echo "==> using: $($PY -c 'import sys; print(sys.executable)')"

echo "==> install PyTorch 2.6.x (overwrites newer torch if present)"
"$PY" -m pip install --upgrade "torch>=2.6.0,<2.7.0" --force-reinstall

echo "==> PyG wheels matching torch (scatter, pyg-lib, sparse, cluster, …)"
"$PY" "$ROOT/desktop/prep_pyg_stack.py" --install

echo "==> verify imports"
"$PY" "$ROOT/desktop/verify_torch_stack.py"

echo "==> torch version:"
"$PY" -c "import torch; print(torch.__version__)"

echo "OK — PyTorch 2.6 + PyG aligned. You can: bash scripts/build_macos_desktop.sh"
