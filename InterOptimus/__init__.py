"""InterOptimus: interface optimization workflows."""

from __future__ import annotations

import os


def _configure_matplotlib_defaults() -> None:
    """Turn on tight layout for most figures; stereographic helpers opt out per-figure."""
    if os.environ.get("INTEROPTIMUS_NO_AUTO_LAYOUT", "").lower() in ("1", "true", "yes"):
        return
    import matplotlib as mpl

    mpl.rcParams["figure.autolayout"] = True


_configure_matplotlib_defaults()
