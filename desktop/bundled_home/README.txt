Portable "home" for InterOptimus Desktop
==========================================

Eqnorm / MLIP checkpoints ship **inside** the PyInstaller app under:

  .cache/InterOptimus/checkpoints/eqnorm*.pt

Put the Eqnorm weight file here **or** under `~/.cache/InterOptimus/checkpoints/` before
running `pyinstaller desktop/interoptimus_desktop.spec`. The spec copies from the user
cache into this folder when packaging, then embeds the whole `bundled_home/` tree in
`InterOptimus.app`.

See https://github.com/yzchen08/eqnorm for the model and dependencies.

You may commit the bundled Eqnorm model to this path for release builds if desired.
