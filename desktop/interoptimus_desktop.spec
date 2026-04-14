# -*- mode: python ; coding: utf-8 -*-
# Build from repo root. The MatRIS checkpoint file below is **copied into the app bundle**
# (bundled_home → PyInstaller datas). Commit it under desktop/bundled_home/.cache/matris/ for releases.
#
#   pip install -e ".[web,desktop]"
#   pip install git+https://github.com/HPC-AI-Team/MatRIS.git
#   pyinstaller desktop/interoptimus_desktop.spec
#
# Output: dist/interoptimus_desktop/ (folder) and on macOS also dist/InterOptimus.app

import sys
from pathlib import Path

block_cipher = None

# SPECPATH is the directory containing this .spec file (e.g. .../repo/desktop).
repo = Path(SPECPATH).resolve().parent
_MATRIS_CKPT = repo / "desktop" / "bundled_home" / ".cache" / "matris" / "MatRIS_10M_OAM.pth.tar"
if not _MATRIS_CKPT.is_file():
    raise SystemExit(
        "PyInstaller: MatRIS checkpoint is required inside the app bundle. Missing file:\n"
        f"  {_MATRIS_CKPT}\n"
        "Place MatRIS_10M_OAM.pth.tar at that path (it is un-ignored in .gitignore for release commits)."
    )

try:
    from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

    _hooks = True
except ImportError:
    _hooks = False

datas = []
if _hooks:
    try:
        datas += collect_data_files("InterOptimus")
    except Exception:
        pass
    try:
        datas += collect_data_files("matris")
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

hiddenimports = [
    "InterOptimus.web.local_workflow",
    "InterOptimus.desktop_app.worker",
    "matris",
    "matris.applications",
    "matris.applications.base",
    "jobflow",
    "atomate2",
]

a = Analysis(
    [str(repo / "desktop" / "pyinstaller_main.py")],
    pathex=[str(repo)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
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
