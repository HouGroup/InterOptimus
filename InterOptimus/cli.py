"""Command line entry point for InterOptimus convenience commands."""

from __future__ import annotations

import sys


def main() -> None:
    """Dispatch ``itom`` subcommands."""
    argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: itom <command> [options]\n\n"
            "commands:\n"
            "  config       configure jobflow, jobflow-remote, atomate2, MongoDB, and checkpoints\n"
            "  checkpoints  download or verify MLIP checkpoints\n"
            "  doctor       check what is already configured and what is missing\n\n"
            "Run `itom config --help` for configuration options."
        )
        return

    command = argv[0]
    if command == "config":
        from InterOptimus.deploy_jobflow_stack import main as config_main

        sys.argv = [f"{sys.argv[0]} config", *argv[1:]]
        config_main()
        return
    if command == "checkpoints":
        from InterOptimus.checkpoints import main as checkpoints_main

        sys.argv = [f"{sys.argv[0]} checkpoints", *argv[1:]]
        checkpoints_main()
        return
    if command == "doctor":
        from InterOptimus.doctor import main as doctor_main

        sys.argv = [f"{sys.argv[0]} doctor", *argv[1:]]
        doctor_main()
        return

    print(f"ERROR: unknown command: {command}", file=sys.stderr)
    print("Run `itom --help` for available commands.", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
