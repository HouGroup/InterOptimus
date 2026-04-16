"""
InterOptimus agent utilities for **workflow execution** (no natural-language LLM in this branch).

Primary entry points:

- :mod:`InterOptimus.agents.simple_iomaker` — JSON/YAML config → submit / run IOMaker
- :mod:`InterOptimus.agents.iomaker_job` — programmatic BuildConfig + ``execute_iomaker_from_settings``
- :mod:`InterOptimus.agents.remote_submit` — progress / fetch results after server submit
"""
