"""
CLI entry for the InterOptimus web UI (FastAPI + uvicorn).
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    try:
        import uvicorn
    except ModuleNotFoundError:
        print(
            "The web UI requires optional dependencies. Install with:\n"
            "  pip install 'InterOptimus[web]'\n"
            "or:\n"
            "  pip install fastapi uvicorn[standard] python-multipart",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    p = argparse.ArgumentParser(description="InterOptimus IOMaker web UI (browser)")
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    p.add_argument(
        "--reload",
        action="store_true",
        help="Dev-only: auto-reload on code changes (do not use in production)",
    )
    args = p.parse_args(argv)

    uvicorn.run(
        "InterOptimus.web_app.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=False,
    )


if __name__ == "__main__":
    main()
