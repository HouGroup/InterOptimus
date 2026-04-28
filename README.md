
<img width="2752" height="1536" alt="InterOptimus" src="https://github.com/user-attachments/assets/0934d4f7-56d5-41bd-b71d-c1137920d19e" />

# InterOptimus

Crystal **interface** search and optimization with MLIP acceleration, optional VASP, and **Jobflow** / **jobflow-remote** execution.

This distribution focuses on the **JSON-driven IOMaker pipeline** (submit → lattice match → MLIP → optional DFT → reports and fetch tooling).

## Highlights

- Lattice matching, termination screening, single / double interface builds
- MLIP global minimization; optional VASP (cluster workers)
- **One-shot config**: `interoptimus-simple -c your.yaml` or `run_simple_iomaker(...)`
- Outputs include `io_flow.json`, `io_report.txt`, `pairs_summary.txt`, `opt_results.pkl` (use `iomaker_fetch_results` / `remote_submit` helpers after server submit)

## Installation

From PyPI (after publish):

```bash
# Core workflow (lattice match, jobflow, interface pipeline):
pip install InterOptimus

# Optional browser UI:
pip install "InterOptimus[web]"   # interoptimus-web (FastAPI + Plotly); PyYAML is in core deps
```

From a git checkout:

```bash
pip install -e .
```

See **[GETTING_STARTED](https://github.com/HouGroup/InterOptimus/blob/HEAD/docs/GETTING_STARTED.md)** for first-time server setup (MongoDB, jobflow-remote, POTCAR, MLIP checkpoints). Links use `HEAD` so they stay valid on the PyPI project description and on GitHub.

## Quick start (recommended)

```bash
# 1) Copy and edit paths / cluster / workers
cp InterOptimus/agents/simple_iomaker.example.json my_run.json
vim my_run.json

# 2) On the login node, after JOBFLOW / JFREMOTE / conda are configured:
interoptimus-simple -c my_run.json
```

Python API:

```python
from pathlib import Path
import json
from InterOptimus.agents.simple_iomaker import run_simple_iomaker
from InterOptimus.agents.remote_submit import iomaker_fetch_results, iomaker_status

with open("my_run.json") as f:
    result = run_simple_iomaker(json.load(f))

print(iomaker_status(result))
# When finished:
# iomaker_fetch_results(dest_dir=Path("./out"), result=result)
```

Lower-level builder (already-normalized `settings` dict):

```python
from InterOptimus.agents.iomaker_job import (
    LocalBuildConfig,
    execute_iomaker_from_settings,
    normalize_iomaker_settings_from_full_dict,
)
```

Full parameter reference: **[simple_iomaker_parameters](https://github.com/HouGroup/InterOptimus/blob/HEAD/docs/simple_iomaker_parameters.md)**.

### Optional: browser GUI (local runs)

```bash
pip install -e '.[web]'
interoptimus-web
```

The web UI drives the same `run_simple_iomaker` pipeline as the CLI; session directories default to `~/.interoptimus/web_sessions/` (override with `INTEROPTIMUS_DESKTOP_SESSIONS` or `INTEROPTIMUS_WEB_SESSIONS`). Optional relaxation telemetry uses `INTEROPTIMUS_VIZ_LOG` + `INTEROPTIMUS_VIZ_ENABLE` (see `InterOptimus.viz_runtime`).

## Requirements

- Python ≥ 3.10
- Core stack: `pymatgen`, `interfacemaster`, `atomate2`, `jobflow`, `jobflow-remote`, `qtoolkit`, … (see `setup.py`)
- MLIP stack (optional): provision conda envs and jobflow-remote workers with **`itom config --with-mlip-workers`** (see **[GETTING_STARTED](docs/GETTING_STARTED.md)**), or install `torch`, `orb-models`, `sevenn`, and `deepmd-kit` manually into the Python env that runs MLIP. **MatRIS** is not on PyPI; install from the upstream project if you use `calc=matris`.

## Layout

```
InterOptimus/
├── itworker.py       # InterfaceWorker (physics + optimization)
├── matching.py       # Lattice matching & screening
├── jobflow.py        # IOMaker + Jobflow makers
├── mlip.py           # MLIP calculators + checkpoint resolution
├── agents/
│   ├── simple_iomaker.py   # CLI + run_simple_iomaker / status / fetch / rerun helpers
│   ├── iomaker_job.py      # BuildConfig + execute_iomaker_from_settings
│   ├── remote_submit.py    # submit_io_flow_locally, progress, fetch
│   └── server_env.py       # interoptimus-env
├── web_app/                # FastAPI UI + job_worker (interoptimus-web)
├── session_workflow.py     # web session driver (form → run_simple_iomaker)
├── result_bundle.py        # curated result/ artifact folder
└── docs/
```

## License

MIT — see `LICENSE`.
