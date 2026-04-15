# InterOptimus — Crystal interface optimization

Efficient Python workflows for interface simulation: lattice matching, structure building, and MLIP-based relaxation (Eqnorm in this fork).

## Core features

1. Lattice matching visualized by polar projection
2. Symmetry analysis to screen duplicate matches and terminations
3. Interface energy exploration with MLIP (Eqnorm) and optional reporting

## Desktop IOMaker

- **`agents/iomaker_core.py`** — build `io_flow.json` and run locally via Jobflow `run_locally`
- **`agents/simple_iomaker.py`** — JSON/YAML CLI (`interoptimus-simple`)
- **`desktop_app/`** — Tkinter GUI (`interoptimus-desktop`)

Install:

```bash
pip install .
```
