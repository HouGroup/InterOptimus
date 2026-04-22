"""
Allow: python -m InterOptimus.web_app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import sys

from InterOptimus.web_app.cli import main

if __name__ == "__main__":
    main(sys.argv[1:])
