# -*- mode: python ; coding: utf-8 -*-
# Build from repo root. Eqnorm weights are staged into bundled_home (then embedded via datas).
# Put eqnorm*.pt under ~/.cache/InterOptimus/checkpoints/ and/or
# desktop/bundled_home/.cache/InterOptimus/checkpoints/ — the spec copies from the user cache
# into bundled_home when newer, so the bundle does not rely on a download at runtime.
#
#   pip install -e ".[web,desktop]"
#   # Eqnorm needs torch_scatter built for THIS torch (see desktop/verify_torch_stack.py):
#   # PyG wheel URLs use torch-X.Y.Z+cpu.html when torch.__version__ has no +local tag (see desktop/pyg_wheel_url.py).
#   python desktop/prep_pyg_stack.py --install
#   python desktop/verify_torch_stack.py
#   pyinstaller desktop/interoptimus_desktop.spec
#
# Output: dist/interoptimus_desktop/ (folder) and on macOS also dist/InterOptimus.app

import shutil
import sys
from pathlib import Path

block_cipher = None

repo = Path(SPECPATH).resolve().parent
_MLIP_CACHE = repo / "desktop" / "bundled_home" / ".cache" / "InterOptimus" / "checkpoints"
_USER_MLIP_CACHE = Path.home() / ".cache" / "InterOptimus" / "checkpoints"


def _stage_eqnorm_checkpoints_for_bundle() -> None:
    """
    Copy eqnorm weights from the developer's ~/.cache/InterOptimus/checkpoints/ into
    bundled_home so PyInstaller embeds them (no runtime download). Repo-local files in
    _MLIP_CACHE are kept unless the user-cache copy is newer.
    """
    _MLIP_CACHE.mkdir(parents=True, exist_ok=True)
    if not _USER_MLIP_CACHE.is_dir():
        return
    for src in sorted(_USER_MLIP_CACHE.glob("eqnorm*.pt")) + sorted(
        _USER_MLIP_CACHE.glob("eqnorm*.pth")
    ):
        dst = _MLIP_CACHE / src.name
        try:
            if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
                continue
        except OSError:
            pass
        shutil.copy2(src, dst)


_stage_eqnorm_checkpoints_for_bundle()
_EQN_CKPTS = sorted(_MLIP_CACHE.glob("eqnorm*.pt")) + sorted(_MLIP_CACHE.glob("eqnorm*.pth"))
if not _EQN_CKPTS:
    raise SystemExit(
        "PyInstaller: an Eqnorm checkpoint is required inside the app bundle. Missing under:\n"
        f"  {_MLIP_CACHE}\n"
        f"and not found under the user cache (used when packaging):\n"
        f"  {_USER_MLIP_CACHE}\n"
        "Place eqnorm*.pt (or .pth) in either location before packaging. "
        "See https://github.com/yzchen08/eqnorm"
    )

try:
    from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

    _hooks = True
except ImportError:
    _hooks = False

try:
    from PyInstaller.utils.hooks import collect_submodules
except ImportError:
    collect_submodules = None  # type: ignore

# Editable (PEP 660) installs often live outside site-packages; PyInstaller must see that root.
import importlib.util

_im = importlib.util.find_spec("interfacemaster")
if _im is None or not getattr(_im, "origin", None):
    raise SystemExit(
        "PyInstaller: `interfacemaster` is not importable in this Python.\n"
        "Install it in the build env, e.g. `pip install interfacemaster` or `pip install -e /path/to/interface_master`."
    )
_interfacemaster_pathex = str(Path(_im.origin).resolve().parent.parent)

datas = []
if _hooks:
    try:
        datas += collect_data_files("InterOptimus")
    except Exception:
        pass
    try:
        datas += collect_data_files("pymatgen")
    except Exception:
        pass
    try:
        datas += collect_data_files("interfacemaster")
    except Exception:
        pass
    try:
        datas += collect_data_files("eqnorm")
    except Exception:
        pass
    # qtoolkit/io/base.py reads scheduler header templates at import time (PBSIO etc.).
    try:
        datas += collect_data_files("qtoolkit")
    except Exception:
        pass

_bh = repo / "desktop" / "bundled_home"
if not _bh.is_dir():
    raise SystemExit(f"PyInstaller: missing desktop bundle directory: {_bh}")
datas.append((str(_bh), "bundled_home"))

binaries = []
if _hooks:
    try:
        binaries += collect_dynamic_libs("torch")
    except Exception:
        pass
    # PyG native extensions (eqnorm → torch_geometric); keep libs consistent with collect_dynamic_libs("torch").
    for _pkg in ("torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv", "torch_geometric"):
        try:
            binaries += collect_dynamic_libs(_pkg)
        except Exception:
            pass

hiddenimports = [
    "InterOptimus.web.local_workflow",
    "InterOptimus.desktop_app.worker",
    "eqnorm",
    "eqnorm.calculator",
    # EqnormCalculator does importlib.import_module(f"{model_name}.{model_variant}") e.g. eqnorm.eqnorm-mptrj
    "eqnorm.eqnorm-mptrj",
    "jobflow",
    "atomate2",
    "interfacemaster",
    "interfacemaster.cellcalc",
    "interfacemaster.hetero_searching",
    # deepmd → array_api_compat uses dynamic __import__(package + ".fft")
    "array_api_compat.numpy.fft",
    "array_api_compat.numpy.linalg",
]
# torch_scatter: see desktop/hooks/hook-torch_scatter.py (native extensions + noarchive below).
if _hooks and collect_submodules:
    try:
        hiddenimports += collect_submodules("array_api_compat")
    except Exception:
        pass
    try:
        hiddenimports += collect_submodules("eqnorm")
    except Exception:
        pass
    try:
        hiddenimports += collect_submodules("torch_scatter")
    except Exception:
        pass

a = Analysis(
    [str(repo / "desktop" / "pyinstaller_main.py")],
    pathex=[str(repo), _interfacemaster_pathex],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(repo / "desktop" / "hooks")],
    hooksconfig={},
    # Load libtorch before torch_scatter (see desktop/hooks/rthook_torch_before_scatter.py).
    runtime_hooks=[str(repo / "desktop" / "hooks" / "rthook_torch_before_scatter.py")],
    excludes=[],
    # Extract bytecode to the filesystem so packages with C extensions (torch_scatter, …) stay
    # next to their .so files; avoids PyTorch "Could not find module '_version_cpu'" in .app.
    noarchive=True,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# macOS: console=False so double-clicking the .app from Finder gets a normal GUI launch;
# console=True console apps often show no window and look like "nothing happened" on failure.
_console = False if sys.platform == "darwin" else True

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="interoptimus_desktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=_console,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="interoptimus_desktop",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="InterOptimus.app",
        icon=None,
        bundle_identifier="edu.tsinghua.interoptimus.desktop",
        info_plist={
            "CFBundleDisplayName": "InterOptimus",
            "CFBundleName": "InterOptimus",
            "NSHighResolutionCapable": True,
        },
    )
