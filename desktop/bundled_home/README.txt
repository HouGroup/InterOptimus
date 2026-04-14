Portable "home" for InterOptimus Desktop
==========================================

MatRIS weights ship **inside** the PyInstaller app under:

  .cache/matris/MatRIS_10M_OAM.pth.tar

Put that file here before running `pyinstaller desktop/interoptimus_desktop.spec`.
The spec requires it and embeds the whole `bundled_home/` tree in `InterOptimus.app`.

You may commit `MatRIS_10M_OAM.pth.tar` to this path (see root `.gitignore` exception) for release builds.
