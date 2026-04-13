# Getting started on your own server

This guide lists what a **new administrator** must have in place before `interoptimus-simple` can run end-to-end (especially **server / cluster** mode).

## 1. Python environment

- Use **Python 3.10+** (3.12 is fine).
- Avoid running loose scripts **from inside** `InterOptimus/InterOptimus/` with `python ./something.py`: that directory contains a file named `jobflow.py`, which can **shadow** the real `jobflow` package and cause circular import errors. Prefer `python -m …` from the repo root or `pip install -e .` and import `InterOptimus.*` from a neutral working directory.
- Install InterOptimus in editable mode from a checkout:

  ```bash
  git clone <your-fork-or-upstream-url> InterOptimus
  cd InterOptimus
  git checkout pure-workflow   # if you maintain this branch on the remote
  pip install -e .
  ```

- If `pip install` pulls MLIP wheels that expect **MPI** (e.g. some `deepmd-kit` builds), install the small **`mpich`** PyPI shim or use a conda MPI stack that matches your site policy.

## 2. Jobflow + MongoDB

- **atomate2 / jobflow** store workflow state in **MongoDB**. You need a reachable `host:port`, database name, and credentials.
- Write **`~/.jobflow.yaml`** and **`~/.atomate2.yaml`** (or set `JOBFLOW_CONFIG_FILE` / `ATOMATE2_CONFIG_FILE`). On clusters, use a hostname that **compute nodes** can resolve (not only `localhost` unless workers run on the login node).

## 3. jobflow-remote

- Configure **`~/.jfremote/<project>.yaml`** with at least one **worker** whose `pre_run` activates the conda env that has InterOptimus + MLIP + (if needed) VASP tooling.
- Start a **runner** on a machine that can execute those jobs (`jf runner start` — see jobflow-remote docs).
- Set **`JFREMOTE_PROJECT`** (and optionally `JFREMOTE_PROJECTS_FOLDER`) in the shell where you submit.

## 4. MLIP checkpoints

- Place ORB / SevenNet / DPA checkpoints under **`~/.cache/InterOptimus/checkpoints/`**, or set **`INTEROPTIMUS_CHECKPOINT_DIR`** to an absolute path visible on **login and compute** nodes.
- **MatRIS** loads weights from **`~/.cache/matris/`** by library convention; symlink or copy your `MatRIS_10M_OAM.pth.tar` there if you use MatRIS.

## 5. VASP (only if `do_vasp: true`)

- **POTCAR**: set **`PMG_VASP_PSP_DIR`** (pymatgen) to your POTCAR library root.
- **Binaries / modules**: encode `module load …` (or `PATH`) in the worker’s **`pre_run`** or in **`vasp_exec_config.pre_run`** in your JSON config.
- Ensure **KPOINTS / INCAR overrides** in the JSON match your cluster policy.

## 6. Sanity checks

```bash
interoptimus-env              # prints jf / workers / VASP hints
jf project check --errors   # after editing ~/.jfremote/*.yaml
```

## 7. First workflow file

- Start from **`InterOptimus/agents/simple_iomaker.example.json`**.
- Read **`docs/simple_iomaker_parameters.md`** for every allowed key.
- Run:

  ```bash
  interoptimus-simple -c my_run.json
  ```

If anything fails, capture **`io_report.txt`** (after a local run) or **`jf job info <uuid>`** (after server submit) when asking for support.
