
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

```bash
pip install -e .
```

See **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** for first-time server setup (MongoDB, jobflow-remote, POTCAR, MLIP checkpoints).

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
from InterOptimus.agents.simple_iomaker import (
    run_simple_iomaker,
    iomaker_status,
    iomaker_fetch_results,
)

with open("my_run.json") as f:
    result = run_simple_iomaker(json.load(f))

print(iomaker_status(result))
# When finished:
# iomaker_fetch_results(result, copy_images_to=Path("./out"))
```

Lower-level builder (already-normalized `settings` dict):

```python
from InterOptimus.agents.iomaker_job import (
    LocalBuildConfig,
    execute_iomaker_from_settings,
    normalize_iomaker_settings_from_full_dict,
)
```

Full parameter reference: **[docs/simple_iomaker_parameters.md](docs/simple_iomaker_parameters.md)**.

### Optional: Tk desktop GUI (local runs)

The [`server`](https://github.com/HouGroup/InterOptimus/tree/server) branch ships a **native Tk** workflow (no browser). It is included in this package but **not** required for `interoptimus-simple` or cluster submit/fetch.

```bash
# Optional: embed HTML reports in the result pane
pip install -e '.[gui]'

interoptimus-desktop
```

The GUI uses the same `run_simple_iomaker` pipeline as the CLI; session directories default to `~/.interoptimus/web_sessions/` (override with `INTEROPTIMUS_DESKTOP_SESSIONS`, or legacy `INTEROPTIMUS_WEB_SESSIONS`). Optional relaxation telemetry for the live plot uses `INTEROPTIMUS_VIZ_LOG` + `INTEROPTIMUS_VIZ_ENABLE` (see `InterOptimus.viz_runtime`).

## Requirements

- Python ≥ 3.10
- Core stack: `pymatgen`, `interfacemaster`, `atomate2`, `jobflow`, `jobflow-remote`, `qtoolkit`, … (see `setup.py`)
- MLIP stack: `torch` plus the calculator(s) you use (`orb-models`, `sevenn`, `deepmd-kit`, …). **MatRIS** is not on PyPI; install from the upstream project if needed.

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
├── desktop_app/            # optional: interoptimus-desktop (local_workflow.py, gui, …)
└── docs/
```

## License

MIT — see `LICENSE`.
