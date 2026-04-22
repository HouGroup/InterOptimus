# InterOptimus (package overview)

Python toolkit for **film/substrate interface** lattice matching, termination screening, MLIP relaxation / global search, and optional **VASP** workflows via **Jobflow** / **jobflow-remote**.

## Workflow entry (this branch)

- **`agents/simple_iomaker.py`** — `interoptimus-simple` CLI and `run_simple_iomaker` / status / fetch helpers.
- **`agents/iomaker_job.py`** — `BuildConfig`, `execute_iomaker_from_settings`, settings normalization.
- **`agents/remote_submit.py`** — submit to the local host’s jobflow-remote, poll progress, pull artifacts.
- **`web_app/`** — `interoptimus-web` browser UI; **`session_workflow.py`** — same pipeline from HTTP form sessions.

See the repository root **`README.md`** and **`docs/GETTING_STARTED.md`**.

## Core modules

| Module | Role |
|--------|------|
| `itworker.py` | `InterfaceWorker` — physics, MLIP, optimization |
| `matching.py` | Lattice matching & stereographic-style figures |
| `jobflow.py` | `IOMaker` Jobflow makers |
| `mlip.py` | MLIP calculators + checkpoint directory |
| `equi_term.py`, `CNID.py`, `tool.py` | Symmetry / CNID / utilities |

Install from repo root: `pip install -e .`
