# InterOptimus Desktop（仅本机）

## 做什么

- `interoptimus-desktop` 启动 **Tkinter 图形界面**（不再依赖浏览器 / FastAPI）。流程与原先网页版一致：选择 film / substrate CIF、配置参数、调用 `run_simple_iomaker`（默认 [Eqnorm](https://github.com/yzchen08/eqnorm)）。
- 程序把 **HOME / USERPROFILE** 指到便携目录；冻结运行时会把包内 `bundled_home` 里的 MLIP checkpoints 复制到用户可写目录（见 `InterOptimus/desktop_app/entry.py`）。
- **Eqnorm**：需要 `torch`、`torch_geometric`、`vesin` 等（见上游 `pyproject.toml`）。`pip install -e "."` 会从 GitHub 拉取 `eqnorm`。
- **模型文件**：打包前将 `eqnorm*.pt`（或 `.pth`）放在 **`~/.cache/InterOptimus/checkpoints/`** 或 **`desktop/bundled_home/.cache/InterOptimus/checkpoints/`**。运行 `pyinstaller desktop/interoptimus_desktop.spec` 时会自动把用户目录下的 eqnorm 权重复制进 `bundled_home` 并打进 `.app`。未放置时 `EqnormCalculator` 可能从网络下载到 `~/.cache/eqnorm/`（需联网）。

### PyTorch 版本（Eqnorm / 打包推荐）

上游 **eqnorm** 只要求 `torch_geometric` 等，**未锁** PyTorch 大版本。在 macOS 上 **PyTorch 2.11+** 与 PyG 预编译轮子的组合容易出 native 崩溃；**2.6.x** 与 `data.pyg.org` 索引更匹配。建议在**专用 conda 环境**里固定 2.6 再装 PyG：

```bash
conda activate eqnorm
bash desktop/install_eqnorm_torch26.sh
```

或手动：`pip install -r desktop/requirements_eqnorm_torch26.txt --force-reinstall`，再 `python desktop/prep_pyg_stack.py --install`。

可选：`pip install -e ".[eqnorm-torch26]"` 在可编辑安装时约束 `torch` 为 2.6.x。

## 快速试跑（不打包）

```bash
cd /path/to/InterOptimus
pip install -e "."
python -m pip install -e ".[desktop]"
# 可选：将 eqnorm 权重放到 desktop/bundled_home/.cache/InterOptimus/checkpoints/
interoptimus-desktop
```

## 用 PyInstaller 打包（macOS 得到 `.app`）

**Eqnorm 依赖 `torch_scatter`，必须与当前安装的 `torch` 同 ABI 编译。** 打包前请安装 PyG 官方轮子并校验（避免 `.app` 里 `torch_scatter/_version_cpu.so` 启动即崩溃）：

```bash
python desktop/prep_pyg_stack.py --install   # 使用 --no-build-isolation，避免源码包构建时找不到 torch；失败时会跳过 torch-spline-conv 重试
python desktop/verify_torch_stack.py         # 应打印 OK
```

一键（含可编辑安装、对齐 PyG、校验、PyInstaller）：

```bash
conda activate eqnorm   # 或你的环境名：必须先激活，否则可能用到系统/Homebrew 的 python3
bash scripts/build_macos_desktop.sh
```

若仍误用解释器，可指定：`PYTHON=/path/to/conda/envs/eqnorm/bin/python bash scripts/build_macos_desktop.sh`

手动分步：

```bash
python -m pip install -e ".[desktop]"
python desktop/prep_pyg_stack.py --install
python desktop/verify_torch_stack.py
# 必须：将 eqnorm*.pt / .pth 放到 desktop/bundled_home/.cache/InterOptimus/checkpoints/
python -m PyInstaller desktop/interoptimus_desktop.spec
```

产物：

- `dist/interoptimus_desktop/`：目录分发（可执行文件旁有 `bundled_home/`）。
- `dist/InterOptimus.app`：仅在 **macOS** 上构建时生成，双击打开 **GUI**。

Torch 等依赖体积极大，构建耗时且可能需按本机环境微调 `hiddenimports`。

缺少 Eqnorm 模型文件时 `interoptimus_desktop.spec` 会直接报错退出。

## Web 版（可选）

若仍需要浏览器访问的 API，请使用 `pip install -e ".[web]"` 后运行 `interoptimus-web`（见 `InterOptimus/web/app.py`）。
