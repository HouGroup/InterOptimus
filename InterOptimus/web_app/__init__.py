"""
Browser UI for the local ``simple_iomaker`` workflow (same backend as :mod:`InterOptimus.session_workflow`).

Run on a server and open from your laptop via SSH port-forward or LAN::

    interoptimus-web --host 0.0.0.0 --port 8765
    # ssh -L 8765:localhost:8765 user@server
"""

from __future__ import annotations
