
<img width="2752" height="1536" alt="197a99cc-8f8c-4220-9e95-dc3cae8bbc80" src="https://github.com/user-attachments/assets/0934d4f7-56d5-41bd-b71d-c1137920d19e" />

# InterOptimus

Crystal interface optimization toolkit with MLIP acceleration and LLM-assisted workflow building.

InterOptimus helps generate and optimize film/substrate interfaces (single or double), run MLIP or VASP workflows, and produce structured outputs (pairs, reports) for downstream analysis.

## Highlights

- Interface lattice matching and termination screening
- Single or double interface construction with thickness controls
- MLIP-driven global minimization and optional VASP workflows
- Jobflow-based execution (local or remote)
- LLM-assisted workflow generation with natural-language prompts
- Auto outputs: `pairs_best_it/`, `pairs_summary.txt`, `io_report.txt`

## Requirements

- Python >= 3.10 (tested on 3.11)
- Key deps: `pymatgen`, `interfacemaster`, `atomate2`, `jobflow`, `jobflow-remote`, `mp-api`, `qtoolkit`, `ase`
- MLIP deps: `orb-models`, `sevenn`, `deepmd-kit`, `torch`

## Supported MLIPs

- ORB models: https://github.com/orbital-materials/orb-models
- SevenNet: https://github.com/MDIL-SNU/SevenNet
- Deep Potential (DPA): https://github.com/deepmodeling/deepmd-kit

## Installation

```bash
pip install -e .
```

Optional (LLM / remote submission tools):

```bash
pip install -r InterOptimus/agents/requirements_remote.txt
```

## Quick Start

### Core API

```python
from InterOptimus.itworker import InterfaceWorker
from pymatgen.core.structure import Structure

film = Structure.from_file("film.cif")
substrate = Structure.from_file("substrate.cif")

iw = InterfaceWorker(film, substrate)
iw.lattice_matching(max_area=150)
iw.parse_interface_structure_params()
iw.parse_optimization_params(calc="orb-models")
iw.global_minimization()
```

### LLM + IOMaker Workflow

```python
from InterOptimus.agents.llm_iomaker_job import BuildConfig, build_iomaker_flow_from_prompt

cfg = BuildConfig(
    api_key="YOUR_API_KEY",
    base_url="https://api.openai.com/v1",
    submit_target="local",
)

result = build_iomaker_flow_from_prompt(
    "建立双界面模型，厚度为10，最大匹配面积不超过20",
    cfg,
)
```

## Project Layout

```
InterOptimus/
├── itworker.py          # Main interface logic
├── matching.py          # Lattice matching & screening
├── jobflow.py           # Jobflow-based workflows
├── mlip.py              # MLIP calculators
├── CNID.py              # CNID calculations
├── equi_term.py         # Equivalent termination analysis
├── tool.py              # Utility functions
├── agents/              # LLM workflow and remote submission tools
├── Tutorial/            # Example data and notebooks
```

## Outputs

Typical run outputs include:

- `io_flow.json` – serialized jobflow workflow
- `pairs_best_it/` – best interfaces for each pair
- `pairs_summary.txt` – tabulated energies and atom counts
- `io_report.txt` – full report with settings and results

## License


MIT License. See `LICENSE`.

