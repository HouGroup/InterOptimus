# InterOptimus Desktop（仅本机）

## 做什么

- `interoptimus-desktop` 启动 **Tkinter 图形界面**（不再依赖浏览器 / FastAPI）。流程与原先网页版一致：选择 film / substrate CIF、配置参数、调用 `run_simple_iomaker`（MatRIS）。
- 程序把 **HOME / USERPROFILE** 指到便携目录，使 MatRIS 从 **`~/.cache/matris/`** 读权重；冻结运行时权重从包内 `bundled_home` 复制到用户可写目录（见 `InterOptimus/desktop_app/entry.py`）。
- **MatRIS 源码**：`pip install git+https://github.com/HPC-AI-Team/MatRIS.git`；PyInstaller 会把已安装的 `matris` 打进产物。
- **权重**：`MatRIS_10M_OAM.pth.tar` 放在 `desktop/bundled_home/.cache/matris/` 后再打包；`.gitignore` 中已对该文件解除忽略，便于发行提交。

## 快速试跑（不打包）

```bash
cd /path/to/InterOptimus
pip install -e "."
pip install git+https://github.com/HPC-AI-Team/MatRIS.git
# 将 MatRIS_10M_OAM.pth.tar 放到 desktop/bundled_home/.cache/matris/
interoptimus-desktop
```

## 用 PyInstaller 打包（macOS 得到 `.app`）

```bash
pip install -e ".[desktop]"
pip install git+https://github.com/HPC-AI-Team/MatRIS.git
# 必须：将 MatRIS_10M_OAM.pth.tar 放到 desktop/bundled_home/.cache/matris/
pyinstaller desktop/interoptimus_desktop.spec
```

产物：

- `dist/interoptimus_desktop/`：目录分发（可执行文件旁有 `bundled_home/`）。
- `dist/InterOptimus.app`：仅在 **macOS** 上构建时生成，双击打开 **GUI**。

Torch 等依赖体积极大，构建耗时且可能需按本机环境微调 `hiddenimports`。

## 可选

```bash
bash scripts/build_macos_desktop.sh
```

缺少权重文件时 `interoptimus_desktop.spec` 会直接报错退出。

## Web 版（可选）

若仍需要浏览器访问的 API，请使用 `pip install -e ".[web]"` 后运行 `interoptimus-web`（见 `InterOptimus/web/app.py`）。
