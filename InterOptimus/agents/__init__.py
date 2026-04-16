"""
InterOptimus workflow execution: server submit, local run, fetch results.

- :mod:`InterOptimus.agents.simple_iomaker` — JSON/YAML config → submit / run IOMaker
- :mod:`InterOptimus.agents.iomaker_job` — programmatic BuildConfig + ``execute_iomaker_from_settings``
- :mod:`InterOptimus.agents.remote_submit` — progress / fetch results after server submit
- :mod:`InterOptimus.agents.iomaker_core` — small helpers (e.g. MLIP name normalization for the desktop GUI)

Optional Tk desktop (no extra core dependency): ``interoptimus-desktop`` → :mod:`InterOptimus.desktop_app`.
"""
