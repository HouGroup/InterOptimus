"""
Native Tkinter GUI for the Eqnorm ``simple_iomaker`` workflow (no browser).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple

from InterOptimus.web.local_workflow import sessions_root

# ---------------------------------------------------------------------------
# macOS: suppress harmless Tk/Cocoa stderr noise (TSM / Caps Lock LED, etc.)
# ---------------------------------------------------------------------------

_MACOS_STDERR_NOISE = (
    "TSM AdjustCapsLockLEDForKeyTransitionHandling",
    "_ISSetPhysicalKeyboardCapsLockLED",
)


class _FilteredStderr:
    """Drop known noisy single-line logs from Apple frameworks; forward everything else."""

    __slots__ = ("_real", "_buf")

    def __init__(self, real: TextIO) -> None:
        self._real = real
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line += "\n"
            if any(n in line for n in _MACOS_STDERR_NOISE):
                continue
            self._real.write(line)
        return len(s)

    def flush(self) -> None:
        if self._buf:
            if not any(n in self._buf for n in _MACOS_STDERR_NOISE):
                self._real.write(self._buf)
            self._buf = ""
        self._real.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _silence_macos_tk_stderr_noise() -> None:
    if sys.platform != "darwin":
        return
    if isinstance(sys.stderr, _FilteredStderr):
        return
    sys.stderr = _FilteredStderr(sys.stderr)


# ---------------------------------------------------------------------------
# i18n
# ---------------------------------------------------------------------------

STRINGS: Dict[str, Dict[str, str]] = {
    "zh": {
        "app_title": "InterOptimus · Eqnorm",
        "header_subtitle": "Eqnorm / IOMaker 工作流 · 本地计算",
        "lang": "语言 Language",
        "cif_files": "CIF 文件",
        "film_cif": "薄膜 film.cif",
        "substrate_cif": "基底 substrate.cif",
        "browse": "浏览…",
        "tab_basic": "基本",
        "tab_lm": "点阵匹配",
        "tab_structure": "结构",
        "tab_opt": "优化 (Eqnorm)",
        "tab_advanced": "高级选项",
        "adv_hint": "以下为可选 MLIP / 全局优化参数；留空则使用成本预设与默认值。",
        "adv_ckpt_path": "ckpt_path（Eqnorm *.pt / MLIP 权重）",
        "adv_eqnorm_variant": "Eqnorm model_variant（默认 eqnorm-mptrj）",
        "adv_fmax": "fmax（弛豫力收敛）",
        "adv_discut": "discut（近邻截断，Å）",
        "adv_gd_tol": "gd_tol（梯度下降收敛）",
        "adv_n_calls_density": "n_calls_density（全局采样密度）",
        "adv_strain_E_correction": "strain_E_correction",
        "adv_term_screen_tol": "term_screen_tol（终止筛选）",
        "adv_z_range_lo": "z_range 下限",
        "adv_z_range_hi": "z_range 上限",
        "adv_bo_coord_bin": "BO_coord_bin_size",
        "adv_bo_energy_bin": "BO_energy_bin_size",
        "adv_bo_rms_bin": "BO_rms_bin_size",
        "workflow_name": "工作流名称",
        "mlip_calc": "MLIP（固定 Eqnorm）",
        "cost_preset": "成本预设",
        "double_interface": "双界面模型 (double_interface)",
        "lm_max_area": "max_area",
        "lm_max_length_tol": "max_length_tol",
        "lm_max_angle_tol": "max_angle_tol",
        "lm_film_max_miller": "film_max_miller",
        "lm_substrate_max_miller": "substrate_max_miller",
        "st_film_thickness": "film_thickness",
        "st_substrate_thickness": "substrate_thickness",
        "st_termination_ftol": "termination_ftol",
        "st_vacuum_over_film": "vacuum_over_film",
        "opt_device": "device",
        "opt_steps": "steps",
        "do_mlip_gd": "do_mlip_gd",
        "relax_in_ratio": "relax_in_ratio（单界面）",
        "relax_in_layers": "relax_in_layers（单界面）",
        "fix_film_fraction": "fix_film_fraction（单界面）",
        "fix_substrate_fraction": "fix_substrate_fraction（单界面）",
        "set_relax_film_ang": "set_relax_film_ang（单界面）",
        "set_relax_substrate_ang": "set_relax_substrate_ang（单界面）",
        "single_iface_hint": "以下为单界面模型参数；勾选「双界面」时不可用。",
        "run": "运行计算",
        "stop": "终止计算",
        "open_workdir": "打开工作目录",
        "open_stereo": "打开极图 JPG",
        "open_stereo_html": "打开交互极图 HTML",
        "open_zip": "打开 POSCAR zip",
        "log_title": "日志 / io_report",
        "log_hint": "选择 film / substrate CIF，配置参数后点击「运行计算」。\n",
        "running": "\n--- 运行中… ---\n",
        "warn_no_cif": "请选择 film.cif 与 substrate.cif。",
        "err_bad_path": "CIF 文件路径无效。",
        "done_title": "完成",
        "done_msg": "计算完成。",
        "fail_worker": "子进程失败",
        "cancelled": "\n--- 已终止 ---\n",
        "file_na": "文件不可用。",
        "folder_na": "文件夹不可用。",
        "viz_enable": "实时可视化（贝叶斯 + 结构优化，需 matplotlib）",
    },
    "en": {
        "app_title": "InterOptimus · Eqnorm",
        "header_subtitle": "Eqnorm / IOMaker workflow · local run",
        "lang": "Language 语言",
        "cif_files": "CIF files",
        "film_cif": "Film film.cif",
        "substrate_cif": "Substrate substrate.cif",
        "browse": "Browse…",
        "tab_basic": "Basic",
        "tab_lm": "Lattice match",
        "tab_structure": "Structure",
        "tab_opt": "Optimization (Eqnorm)",
        "tab_advanced": "Advanced",
        "adv_hint": "Optional MLIP / global optimization parameters; leave blank to use cost preset defaults.",
        "adv_ckpt_path": "ckpt_path (Eqnorm *.pt / MLIP checkpoint)",
        "adv_eqnorm_variant": "Eqnorm model_variant (default eqnorm-mptrj)",
        "adv_fmax": "fmax (relaxation)",
        "adv_discut": "discut (neighbor cutoff, Å)",
        "adv_gd_tol": "gd_tol (GD convergence)",
        "adv_n_calls_density": "n_calls_density (global sampling)",
        "adv_strain_E_correction": "strain_E_correction",
        "adv_term_screen_tol": "term_screen_tol",
        "adv_z_range_lo": "z_range min",
        "adv_z_range_hi": "z_range max",
        "adv_bo_coord_bin": "BO_coord_bin_size",
        "adv_bo_energy_bin": "BO_energy_bin_size",
        "adv_bo_rms_bin": "BO_rms_bin_size",
        "workflow_name": "Workflow name",
        "mlip_calc": "MLIP (Eqnorm only)",
        "cost_preset": "Cost preset",
        "double_interface": "Double-interface model",
        "lm_max_area": "max_area",
        "lm_max_length_tol": "max_length_tol",
        "lm_max_angle_tol": "max_angle_tol",
        "lm_film_max_miller": "film_max_miller",
        "lm_substrate_max_miller": "substrate_max_miller",
        "st_film_thickness": "film_thickness",
        "st_substrate_thickness": "substrate_thickness",
        "st_termination_ftol": "termination_ftol",
        "st_vacuum_over_film": "vacuum_over_film",
        "opt_device": "device",
        "opt_steps": "steps",
        "do_mlip_gd": "do_mlip_gd",
        "relax_in_ratio": "relax_in_ratio (single-interface)",
        "relax_in_layers": "relax_in_layers (single-interface)",
        "fix_film_fraction": "fix_film_fraction (single-interface)",
        "fix_substrate_fraction": "fix_substrate_fraction (single-interface)",
        "set_relax_film_ang": "set_relax_film_ang (single-interface)",
        "set_relax_substrate_ang": "set_relax_substrate_ang (single-interface)",
        "single_iface_hint": "Single-interface-only options (disabled when double-interface is on).",
        "run": "Run",
        "stop": "Stop",
        "open_workdir": "Open workdir",
        "open_stereo": "Open stereographic JPG",
        "open_stereo_html": "Open interactive stereo HTML",
        "open_zip": "Open POSCAR zip",
        "log_title": "Log / io_report",
        "log_hint": "Choose film / substrate CIF files, set parameters, then click Run.\n",
        "running": "\n--- Running… ---\n",
        "warn_no_cif": "Please select film.cif and substrate.cif.",
        "err_bad_path": "Invalid CIF path.",
        "done_title": "Done",
        "done_msg": "Calculation finished.",
        "fail_worker": "Worker failed",
        "cancelled": "\n--- Stopped ---\n",
        "file_na": "File not available.",
        "folder_na": "Folder not available.",
        "viz_enable": "Live visualization (BO + relax; requires matplotlib)",
    },
}


def _bool_to_str(v: bool) -> str:
    return "true" if v else "false"


class InterOptimusGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self._lang = "zh"
        self._proc: Optional[subprocess.Popen[str]] = None
        self._proc_lock = threading.Lock()
        self._config_temp: Optional[str] = None

        self._film = tk.StringVar()
        self._sub = tk.StringVar()
        self._workflow = tk.StringVar(value="IO_web_eqnorm")
        self._cost = tk.StringVar(value="medium")
        self._double_if = tk.BooleanVar(value=True)

        self._lm_area = tk.StringVar(value="20")
        self._lm_ltol = tk.StringVar(value="0.03")
        self._lm_atol = tk.StringVar(value="0.03")
        self._lm_fmill = tk.StringVar(value="3")
        self._lm_smill = tk.StringVar(value="3")

        self._st_ft = tk.StringVar(value="10")
        self._st_st = tk.StringVar(value="10")
        self._st_ftol = tk.StringVar(value="0.15")
        self._st_vac = tk.StringVar(value="5")

        self._opt_dev = tk.StringVar(value="cpu")
        self._opt_steps = tk.StringVar(value="500")
        self._opt_gd = tk.BooleanVar(value=False)
        self._opt_rir = tk.BooleanVar(value=True)
        self._opt_ril = tk.BooleanVar(value=False)
        self._opt_fff = tk.StringVar(value="0.5")
        self._opt_fs = tk.StringVar(value="0.5")
        self._opt_srf = tk.StringVar(value="0")
        self._opt_srs = tk.StringVar(value="0")

        self._adv_ckpt = tk.StringVar(value="")
        self._adv_eqnorm_variant = tk.StringVar(value="")
        self._adv_fmax = tk.StringVar(value="")
        self._adv_discut = tk.StringVar(value="")
        self._adv_gd_tol = tk.StringVar(value="")
        self._adv_n_calls = tk.StringVar(value="")
        self._adv_strain_E = tk.StringVar(value="default")
        self._adv_term_screen = tk.StringVar(value="")
        self._adv_z_lo = tk.StringVar(value="")
        self._adv_z_hi = tk.StringVar(value="")
        self._adv_bo_coord = tk.StringVar(value="")
        self._adv_bo_energy = tk.StringVar(value="")
        self._adv_bo_rms = tk.StringVar(value="")

        self._viz_enable = tk.BooleanVar(value=True)
        self._viz_poll_active = False
        self._viz_win: Any = None
        self._viz_chk: Optional[ttk.Checkbutton] = None

        self._log: tk.Text
        self._last_payload: Optional[Dict[str, Any]] = None
        self._run_btn: Optional[ttk.Button] = None
        self._stop_btn: Optional[ttk.Button] = None
        self._lang_combo: Optional[ttk.Combobox] = None
        self._i18n_widgets: List[Tuple[Any, str, Callable[..., None]]] = []
        self._notebook: Optional[ttk.Notebook] = None
        self._single_iface_widgets: List[tk.Widget] = []
        self._single_iface_labels: List[tk.Widget] = []
        self._hint_label_opt: Optional[ttk.Label] = None
        self._lf_log: Optional[ttk.LabelFrame] = None
        self._user_cancelled = False
        self._lang_lbl: Optional[ttk.Label] = None

        self._setup_style()
        self._build()
        self._double_if.trace_add("write", lambda *_: self._sync_single_interface_state())
        self._sync_single_interface_state()

    def t(self, key: str) -> str:
        return STRINGS.get(self._lang, STRINGS["zh"]).get(key, key)

    def _setup_style(self) -> None:
        self.root.title(self.t("app_title"))
        self.root.minsize(780, 600)
        # Light theme: cool slate + one accent (calm, readable)
        self._bg = "#eef2f7"
        self._panel = "#ffffff"
        self._accent = "#2563eb"
        self._accent_hover = "#1d4ed8"
        self._muted = "#64748b"
        self._text = "#0f172a"
        self.root.configure(bg=self._bg)

        if sys.platform == "darwin":
            self._font_ui = (".AppleSystemUIFont", 13)
            self._font_ui_sm = (".AppleSystemUIFont", 11)
            self._font_mono = ("Menlo", 11)
        else:
            self._font_ui = ("Segoe UI", 10)
            self._font_ui_sm = ("Segoe UI", 9)
            self._font_mono = ("Consolas", 10)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Main.TFrame", background=self._bg)
        style.configure(
            "Card.TLabelframe",
            background=self._panel,
            relief="flat",
            borderwidth=1,
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=self._panel,
            foreground=self._muted,
            font=self._font_ui_sm + ("bold",),
        )
        style.configure("TLabel", background=self._bg, foreground=self._text, font=self._font_ui)
        style.configure("Card.TLabel", background=self._panel, foreground="#334155", font=self._font_ui)
        style.configure("TEntry", fieldbackground="#ffffff", insertwidth=1)
        style.configure("Accent.TButton", foreground="#ffffff", padding=(18, 8), font=self._font_ui)
        style.map(
            "Accent.TButton",
            background=[("active", self._accent_hover), ("!disabled", self._accent)],
        )
        style.configure("Stop.TButton", foreground="#ffffff", padding=(14, 8), font=self._font_ui)
        style.map("Stop.TButton", background=[("active", "#b91c1c"), ("!disabled", "#dc2626")])
        style.configure(
            "Muted.TButton",
            foreground="#334155",
            padding=(12, 6),
            font=self._font_ui_sm,
        )
        style.map(
            "Muted.TButton",
            background=[("active", "#e2e8f0"), ("!disabled", "#f1f5f9")],
        )
        style.configure("TCheckbutton", background=self._bg, foreground=self._text, font=self._font_ui)
        style.configure("TCombobox", font=self._font_ui)

        style.configure("TNotebook", background=self._bg, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#e8edf4",
            foreground="#475569",
            padding=(14, 8),
            font=self._font_ui_sm,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self._panel), ("!selected", "#e8edf4")],
            foreground=[("selected", self._text), ("!selected", "#64748b")],
        )

        try:
            self._title_font = tkfont.Font(family=self._font_ui[0], size=20, weight="bold")
        except tk.TclError:
            self._title_font = tkfont.Font(size=20, weight="bold")
        try:
            self._subtitle_font = tkfont.Font(family=self._font_ui[0], size=11)
        except tk.TclError:
            self._subtitle_font = tkfont.Font(size=11)

    def _build(self) -> None:
        pad = {"padx": 12, "pady": 6}
        outer = ttk.Frame(self.root, style="Main.TFrame")
        outer.pack(fill="both", expand=True)

        # Header
        hdr = ttk.Frame(outer, style="Main.TFrame")
        hdr.pack(fill="x", padx=12, pady=(14, 8))
        hdr_left = ttk.Frame(hdr, style="Main.TFrame")
        hdr_left.pack(side="left", fill="x", expand=True)
        title = tk.Label(
            hdr_left,
            text=self.t("app_title"),
            font=self._title_font,
            bg=self._bg,
            fg=self._text,
            anchor="w",
        )
        title.pack(anchor="w")
        self._i18n_widgets.append((title, "app_title", lambda w, s: w.config(text=s)))
        sub = tk.Label(
            hdr_left,
            text=self.t("header_subtitle"),
            font=self._subtitle_font,
            bg=self._bg,
            fg=self._muted,
            anchor="w",
        )
        sub.pack(anchor="w", pady=(4, 0))
        self._i18n_widgets.append((sub, "header_subtitle", lambda w, s: w.config(text=s)))

        lang_fr = ttk.Frame(hdr, style="Main.TFrame")
        lang_fr.pack(side="right")
        self._lang_lbl = ttk.Label(lang_fr, text=self.t("lang"))
        self._lang_lbl.pack(side="left", padx=(0, 6))
        self._i18n_widgets.append((self._lang_lbl, "lang", lambda w, s: w.config(text=s)))
        self._lang_combo = ttk.Combobox(
            lang_fr,
            width=8,
            state="readonly",
            values=("中文", "English"),
        )
        self._lang_combo.set("中文" if self._lang == "zh" else "English")
        self._lang_combo.pack(side="left")
        self._lang_combo.bind("<<ComboboxSelected>>", self._on_lang_change)

        ttk.Separator(outer, orient="horizontal").pack(fill="x", padx=12, pady=(0, 2))

        # CIF card
        f_top = ttk.LabelFrame(outer, text=self.t("cif_files"), style="Card.TLabelframe")
        f_top.pack(fill="x", padx=12, pady=(8, 6))
        self._i18n_widgets.append((f_top, "cif_files", lambda w, s: w.config(text=s)))

        r1 = ttk.Frame(f_top)
        r1.pack(fill="x", padx=10, pady=6)
        lb1 = ttk.Label(r1, text=self.t("film_cif"), style="Card.TLabel", width=18)
        lb1.pack(side="left")
        self._i18n_widgets.append((lb1, "film_cif", lambda w, s: w.config(text=s)))
        ttk.Entry(r1, textvariable=self._film, width=50).pack(side="left", fill="x", expand=True, padx=4)
        b1 = ttk.Button(r1, text=self.t("browse"), style="Muted.TButton", command=self._browse_film)
        b1.pack(side="left")
        self._i18n_widgets.append((b1, "browse", lambda w, s: w.config(text=s)))

        r2 = ttk.Frame(f_top)
        r2.pack(fill="x", padx=10, pady=(0, 10))
        lb2 = ttk.Label(r2, text=self.t("substrate_cif"), style="Card.TLabel", width=18)
        lb2.pack(side="left")
        self._i18n_widgets.append((lb2, "substrate_cif", lambda w, s: w.config(text=s)))
        ttk.Entry(r2, textvariable=self._sub, width=50).pack(side="left", fill="x", expand=True, padx=4)
        b2 = ttk.Button(r2, text=self.t("browse"), style="Muted.TButton", command=self._browse_sub)
        b2.pack(side="left")
        self._i18n_widgets.append((b2, "browse", lambda w, s: w.config(text=s)))

        # Notebook
        self._notebook = ttk.Notebook(outer)
        self._notebook.pack(fill="both", expand=True, padx=12, pady=(4, 6))

        t_basic = ttk.Frame(self._notebook, style="Main.TFrame")
        self._notebook.add(t_basic, text=self.t("tab_basic"))
        row = 0
        lb_wn = ttk.Label(t_basic, text=self.t("workflow_name"))
        lb_wn.grid(row=row, column=0, sticky="w", padx=10, pady=4)
        self._i18n_widgets.append((lb_wn, "workflow_name", lambda w, s: w.config(text=s)))
        ttk.Entry(t_basic, textvariable=self._workflow, width=42).grid(row=row, column=1, sticky="w", padx=10, pady=4)
        row += 1

        lb_cp = ttk.Label(t_basic, text=self.t("cost_preset"))
        lb_cp.grid(row=row, column=0, sticky="w", padx=10, pady=4)
        self._i18n_widgets.append((lb_cp, "cost_preset", lambda w, s: w.config(text=s)))
        ttk.Combobox(t_basic, textvariable=self._cost, values=("low", "medium", "high"), width=12, state="readonly").grid(
            row=row, column=1, sticky="w", padx=10, pady=4
        )
        row += 1

        ch_di = ttk.Checkbutton(t_basic, text=self.t("double_interface"), variable=self._double_if)
        ch_di.grid(row=row, column=1, sticky="w", padx=10, pady=4)
        self._i18n_widgets.append((ch_di, "double_interface", lambda w, s: w.config(text=s)))
        row += 1

        lb_mc = ttk.Label(t_basic, text=self.t("mlip_calc"))
        lb_mc.grid(row=row, column=0, sticky="w", padx=10, pady=4)
        self._i18n_widgets.append((lb_mc, "mlip_calc", lambda w, s: w.config(text=s)))
        ttk.Label(t_basic, text="eqnorm", foreground=self._muted).grid(row=row, column=1, sticky="w", padx=10, pady=4)
        row += 1

        t_lm = ttk.Frame(self._notebook, style="Main.TFrame")
        self._notebook.add(t_lm, text=self.t("tab_lm"))
        lm_fields = [
            ("lm_max_area", self._lm_area),
            ("lm_max_length_tol", self._lm_ltol),
            ("lm_max_angle_tol", self._lm_atol),
            ("lm_film_max_miller", self._lm_fmill),
            ("lm_substrate_max_miller", self._lm_smill),
        ]
        for i, (key, var) in enumerate(lm_fields):
            lb = ttk.Label(t_lm, text=self.t(key))
            lb.grid(row=i, column=0, sticky="w", padx=10, pady=3)
            self._i18n_widgets.append((lb, key, lambda w, s: w.config(text=s)))
            ttk.Entry(t_lm, textvariable=var, width=22).grid(row=i, column=1, sticky="w", padx=10, pady=3)

        t_st = ttk.Frame(self._notebook, style="Main.TFrame")
        self._notebook.add(t_st, text=self.t("tab_structure"))
        st_fields = [
            ("st_film_thickness", self._st_ft),
            ("st_substrate_thickness", self._st_st),
            ("st_termination_ftol", self._st_ftol),
            ("st_vacuum_over_film", self._st_vac),
        ]
        for i, (key, var) in enumerate(st_fields):
            lb = ttk.Label(t_st, text=self.t(key))
            lb.grid(row=i, column=0, sticky="w", padx=10, pady=3)
            self._i18n_widgets.append((lb, key, lambda w, s: w.config(text=s)))
            ttk.Entry(t_st, textvariable=var, width=22).grid(row=i, column=1, sticky="w", padx=10, pady=3)

        t_op = ttk.Frame(self._notebook, style="Main.TFrame")
        self._notebook.add(t_op, text=self.t("tab_opt"))
        self._hint_label_opt = ttk.Label(t_op, text=self.t("single_iface_hint"), foreground=self._muted, wraplength=520, justify="left")
        self._hint_label_opt.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(6, 2))
        self._i18n_widgets.append((self._hint_label_opt, "single_iface_hint", lambda w, s: w.config(text=s)))

        r = 1
        lb_od = ttk.Label(t_op, text=self.t("opt_device"))
        lb_od.grid(row=r, column=0, sticky="w", padx=10, pady=3)
        self._i18n_widgets.append((lb_od, "opt_device", lambda w, s: w.config(text=s)))
        ttk.Combobox(t_op, textvariable=self._opt_dev, values=("cpu", "cuda", "gpu"), width=12, state="readonly").grid(
            row=r, column=1, sticky="w", padx=10, pady=3
        )
        r += 1
        lb_os = ttk.Label(t_op, text=self.t("opt_steps"))
        lb_os.grid(row=r, column=0, sticky="w", padx=10, pady=3)
        self._i18n_widgets.append((lb_os, "opt_steps", lambda w, s: w.config(text=s)))
        ttk.Entry(t_op, textvariable=self._opt_steps, width=14).grid(row=r, column=1, sticky="w", padx=10, pady=3)
        r += 1

        ch_gd = ttk.Checkbutton(t_op, text=self.t("do_mlip_gd"), variable=self._opt_gd)
        ch_gd.grid(row=r, column=1, sticky="w", padx=10, pady=3)
        self._i18n_widgets.append((ch_gd, "do_mlip_gd", lambda w, s: w.config(text=s)))
        r += 1

        ch_rir = ttk.Checkbutton(t_op, text=self.t("relax_in_ratio"), variable=self._opt_rir)
        ch_rir.grid(row=r, column=1, sticky="w", padx=10, pady=3)
        self._single_iface_widgets.append(ch_rir)
        self._i18n_widgets.append((ch_rir, "relax_in_ratio", lambda w, s: w.config(text=s)))
        r += 1

        ch_ril = ttk.Checkbutton(t_op, text=self.t("relax_in_layers"), variable=self._opt_ril)
        ch_ril.grid(row=r, column=1, sticky="w", padx=10, pady=3)
        self._single_iface_widgets.append(ch_ril)
        self._i18n_widgets.append((ch_ril, "relax_in_layers", lambda w, s: w.config(text=s)))
        r += 1

        for key, var in [
            ("fix_film_fraction", self._opt_fff),
            ("fix_substrate_fraction", self._opt_fs),
            ("set_relax_film_ang", self._opt_srf),
            ("set_relax_substrate_ang", self._opt_srs),
        ]:
            lb = ttk.Label(t_op, text=self.t(key))
            lb.grid(row=r, column=0, sticky="w", padx=10, pady=3)
            self._single_iface_labels.append(lb)
            self._i18n_widgets.append((lb, key, lambda w, s: w.config(text=s)))
            ent = ttk.Entry(t_op, textvariable=var, width=14)
            ent.grid(row=r, column=1, sticky="w", padx=10, pady=3)
            self._single_iface_widgets.append(ent)
            r += 1

        t_adv = ttk.Frame(self._notebook, style="Main.TFrame")
        self._notebook.add(t_adv, text=self.t("tab_advanced"))
        adv_hint = ttk.Label(
            t_adv,
            text=self.t("adv_hint"),
            foreground=self._muted,
            wraplength=520,
            justify="left",
        )
        adv_hint.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(6, 8))
        self._i18n_widgets.append((adv_hint, "adv_hint", lambda w, s: w.config(text=s)))

        adv_rows: List[Tuple[str, tk.StringVar, int]] = [
            ("adv_ckpt_path", self._adv_ckpt, 52),
            ("adv_eqnorm_variant", self._adv_eqnorm_variant, 28),
            ("adv_fmax", self._adv_fmax, 14),
            ("adv_discut", self._adv_discut, 14),
            ("adv_gd_tol", self._adv_gd_tol, 14),
            ("adv_n_calls_density", self._adv_n_calls, 14),
            ("adv_term_screen_tol", self._adv_term_screen, 14),
            ("adv_z_range_lo", self._adv_z_lo, 14),
            ("adv_z_range_hi", self._adv_z_hi, 14),
            ("adv_bo_coord_bin", self._adv_bo_coord, 14),
            ("adv_bo_energy_bin", self._adv_bo_energy, 14),
            ("adv_bo_rms_bin", self._adv_bo_rms, 14),
        ]
        ar = 1
        for key, var, width in adv_rows:
            lb = ttk.Label(t_adv, text=self.t(key))
            lb.grid(row=ar, column=0, sticky="w", padx=10, pady=3)
            self._i18n_widgets.append((lb, key, lambda w, s: w.config(text=s)))
            ttk.Entry(t_adv, textvariable=var, width=width).grid(row=ar, column=1, sticky="ew", padx=10, pady=3)
            ar += 1

        lb_se = ttk.Label(t_adv, text=self.t("adv_strain_E_correction"))
        lb_se.grid(row=ar, column=0, sticky="w", padx=10, pady=3)
        self._i18n_widgets.append((lb_se, "adv_strain_E_correction", lambda w, s: w.config(text=s)))
        ttk.Combobox(
            t_adv,
            textvariable=self._adv_strain_E,
            values=("default", "true", "false"),
            width=12,
            state="readonly",
        ).grid(row=ar, column=1, sticky="w", padx=10, pady=3)
        ar += 1

        t_adv.columnconfigure(1, weight=1)

        f_act = ttk.Frame(outer, style="Main.TFrame")
        f_act.pack(fill="x", padx=12, pady=8)
        self._viz_chk = ttk.Checkbutton(
            f_act,
            text=self.t("viz_enable"),
            variable=self._viz_enable,
        )
        self._viz_chk.pack(side="left", padx=(0, 12))
        self._i18n_widgets.append((self._viz_chk, "viz_enable", lambda w, s: w.config(text=s)))

        self._run_btn = ttk.Button(f_act, text=self.t("run"), style="Accent.TButton", command=self._on_run)
        self._run_btn.pack(side="left", padx=(0, 8))
        self._i18n_widgets.append((self._run_btn, "run", lambda w, s: w.config(text=s)))

        self._stop_btn = ttk.Button(f_act, text=self.t("stop"), style="Stop.TButton", command=self._on_stop, state="disabled")
        self._stop_btn.pack(side="left", padx=(0, 12))
        self._i18n_widgets.append((self._stop_btn, "stop", lambda w, s: w.config(text=s)))

        for key, cmd in [
            ("open_workdir", self._open_workdir),
            ("open_stereo", self._open_stereo),
            ("open_stereo_html", self._open_stereo_html),
            ("open_zip", self._open_zip),
        ]:
            b = ttk.Button(f_act, text=self.t(key), style="Muted.TButton", command=cmd)
            b.pack(side="left", padx=4)
            self._i18n_widgets.append((b, key, lambda w, s: w.config(text=s)))

        self._lf_log = ttk.LabelFrame(outer, text=self.t("log_title"), style="Card.TLabelframe")
        self._lf_log.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._i18n_widgets.append((self._lf_log, "log_title", lambda w, s: w.config(text=s)))

        scroll = ttk.Scrollbar(self._lf_log)
        self._log = tk.Text(
            self._lf_log,
            height=14,
            wrap="word",
            font=self._font_mono,
            bg="#f8fafc",
            fg="#0f172a",
            insertbackground="#2563eb",
            relief="flat",
            padx=10,
            pady=10,
            highlightthickness=0,
        )
        scroll.pack(side="right", fill="y")
        self._log.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        self._log.config(yscrollcommand=scroll.set)
        scroll.config(command=self._log.yview)

        self._log_insert(self.t("log_hint"))
        self._refresh_notebook_tabs()
        self._refresh_i18n_static_labels()

    def _refresh_notebook_tabs(self) -> None:
        if not self._notebook:
            return
        tabs = ["tab_basic", "tab_lm", "tab_structure", "tab_opt", "tab_advanced"]
        for i, tab_id in enumerate(self._notebook.tabs()):
            if i < len(tabs):
                self._notebook.tab(tab_id, text=self.t(tabs[i]))

    def _refresh_i18n_static_labels(self) -> None:
        """Update labels that store keys in _i18n_widgets."""
        for w, key, setter in self._i18n_widgets:
            try:
                setter(w, self.t(key))
            except tk.TclError:
                pass

    def _on_lang_change(self, _evt: Any = None) -> None:
        choice = self._lang_combo.get() if self._lang_combo else "中文"
        self._lang = "zh" if choice.startswith("中") else "en"
        self.root.title(self.t("app_title"))
        self._refresh_notebook_tabs()
        self._refresh_i18n_static_labels()
        self._sync_single_interface_state()

    def _sync_single_interface_state(self) -> None:
        di = self._double_if.get()
        state = "disabled" if di else "normal"
        for w in self._single_iface_widgets:
            try:
                w.configure(state=state)
            except tk.TclError:
                pass
        for w in self._single_iface_labels:
            try:
                w.configure(foreground=self._muted if di else "#334155")
            except tk.TclError:
                pass
        if self._hint_label_opt:
            try:
                self._hint_label_opt.configure(foreground=self._muted)
            except tk.TclError:
                pass

    def _log_insert(self, s: str) -> None:
        self._log.insert("end", s)
        self._log.see("end")

    def _browse_film(self) -> None:
        p = filedialog.askopenfilename(
            title=self.t("film_cif"),
            filetypes=[("CIF", "*.cif *.CIF"), ("All", "*.*")],
        )
        if p:
            self._film.set(p)

    def _browse_sub(self) -> None:
        p = filedialog.askopenfilename(
            title=self.t("substrate_cif"),
            filetypes=[("CIF", "*.cif *.CIF"), ("All", "*.*")],
        )
        if p:
            self._sub.set(p)

    def _collect_form(self) -> Dict[str, Any]:
        return {
            "workflow_name": self._workflow.get(),
            "mlip_calc": "eqnorm",
            "cost_preset": self._cost.get(),
            "double_interface": _bool_to_str(self._double_if.get()),
            "execution": "local",
            "lm_max_area": self._lm_area.get(),
            "lm_max_length_tol": self._lm_ltol.get(),
            "lm_max_angle_tol": self._lm_atol.get(),
            "lm_film_max_miller": self._lm_fmill.get(),
            "lm_substrate_max_miller": self._lm_smill.get(),
            "st_film_thickness": self._st_ft.get(),
            "st_substrate_thickness": self._st_st.get(),
            "st_termination_ftol": self._st_ftol.get(),
            "st_vacuum_over_film": self._st_vac.get(),
            "opt_device": self._opt_dev.get(),
            "opt_steps": self._opt_steps.get(),
            "do_mlip_gd": _bool_to_str(self._opt_gd.get()),
            "relax_in_ratio": _bool_to_str(self._opt_rir.get()),
            "relax_in_layers": _bool_to_str(self._opt_ril.get()),
            "fix_film_fraction": self._opt_fff.get(),
            "fix_substrate_fraction": self._opt_fs.get(),
            "set_relax_film_ang": self._opt_srf.get(),
            "set_relax_substrate_ang": self._opt_srs.get(),
            "adv_ckpt_path": self._adv_ckpt.get(),
            "adv_eqnorm_variant": self._adv_eqnorm_variant.get(),
            "adv_fmax": self._adv_fmax.get(),
            "adv_discut": self._adv_discut.get(),
            "adv_gd_tol": self._adv_gd_tol.get(),
            "adv_n_calls_density": self._adv_n_calls.get(),
            "adv_strain_E_correction": (
                "" if self._adv_strain_E.get() == "default" else self._adv_strain_E.get()
            ),
            "adv_term_screen_tol": self._adv_term_screen.get(),
            "adv_z_range_lo": self._adv_z_lo.get(),
            "adv_z_range_hi": self._adv_z_hi.get(),
            "adv_bo_coord_bin": self._adv_bo_coord.get(),
            "adv_bo_energy_bin": self._adv_bo_energy.get(),
            "adv_bo_rms_bin": self._adv_bo_rms.get(),
        }

    def _on_run(self) -> None:
        fp = self._film.get().strip()
        sp = self._sub.get().strip()
        if not fp or not sp:
            messagebox.showwarning(self.t("app_title"), self.t("warn_no_cif"))
            return
        if not Path(fp).is_file() or not Path(sp).is_file():
            messagebox.showerror(self.t("app_title"), self.t("err_bad_path"))
            return

        with self._proc_lock:
            if self._proc is not None and self._proc.poll() is None:
                return

        if self._viz_win is not None:
            try:
                self._viz_win.destroy()
            except Exception:
                pass
            self._viz_win = None

        form = self._collect_form()
        viz_path: Optional[str] = None
        if self._viz_enable.get():
            try:
                from InterOptimus.desktop_app import viz_window as vzmod

                if getattr(vzmod, "_HAS_MPL", False):
                    vtf = tempfile.NamedTemporaryFile(prefix="io_viz_", suffix=".jsonl", delete=False)
                    vtf.close()
                    viz_path = vtf.name
                    open(viz_path, "w", encoding="utf-8").close()
            except Exception:
                viz_path = None

        fd, path = tempfile.mkstemp(suffix=".json", prefix="io_eqnorm_")
        rfd, worker_result_path = tempfile.mkstemp(suffix=".json", prefix="io_worker_result_")
        os.close(rfd)
        try:
            cfg_obj: Dict[str, Any] = {"film": fp, "sub": sp, "form": form}
            if viz_path:
                cfg_obj["viz_log"] = viz_path
                cfg_obj["viz_enable"] = True
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cfg_obj, f, ensure_ascii=False)
        except OSError as e:
            try:
                os.unlink(worker_result_path)
            except OSError:
                pass
            messagebox.showerror(self.t("app_title"), str(e))
            return

        self._config_temp = path
        self._user_cancelled = False
        self._viz_poll_active = bool(viz_path)
        if viz_path:
            try:
                from InterOptimus.desktop_app.viz_window import WorkflowVizWindow

                self._viz_win = WorkflowVizWindow(
                    self.root,
                    Path(viz_path),
                    is_active=lambda: self._viz_poll_active,
                )
            except Exception as e:
                self._viz_poll_active = False
                self._log_insert(f"\n(可视化不可用: {e})\n")

        self._log_insert(self.t("running"))
        if self._run_btn:
            self._run_btn.config(state="disabled")
        if self._stop_btn:
            self._stop_btn.config(state="normal")

        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--interoptimus-worker", path]
        else:
            cmd = [sys.executable, "-u", "-m", "InterOptimus.desktop_app.worker", path]

        def work() -> None:
            proc: Optional[subprocess.Popen[str]] = None
            try:
                env = os.environ.copy()
                try:
                    from InterOptimus.desktop_app.runtime_env import apply_worker_env

                    apply_worker_env(env)
                except Exception:
                    pass
                env.setdefault("PYTHONUNBUFFERED", "1")
                env["INTEROPTIMUS_WORKER_RESULT"] = worker_result_path
                # Avoid matplotlib opening extra native plot windows in the worker (plt.show / default backend).
                env.setdefault("MPLBACKEND", "Agg")
                if viz_path:
                    env["INTEROPTIMUS_VIZ_LOG"] = viz_path
                    env["INTEROPTIMUS_VIZ_ENABLE"] = "1"
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
                with self._proc_lock:
                    self._proc = proc

                assert proc.stderr is not None
                assert proc.stdout is not None

                def pump_stderr() -> None:
                    for line in iter(proc.stderr.readline, ""):
                        if not line:
                            break
                        self.root.after(0, lambda l=line: self._log_insert(l))

                stdout_parts: list[str] = []

                def pump_stdout() -> None:
                    # Drain stdout while the child runs so native code writing to fd 1 cannot fill
                    # the pipe buffer and deadlock the worker before it emits the final JSON line.
                    while True:
                        chunk = proc.stdout.read(65536)
                        if not chunk:
                            break
                        stdout_parts.append(chunk)

                stderr_thread = threading.Thread(target=pump_stderr, daemon=True)
                stderr_thread.start()
                stdout_thread = threading.Thread(target=pump_stdout, daemon=True)
                stdout_thread.start()

                rc = proc.wait()
                stderr_thread.join(timeout=60.0)
                stdout_thread.join(timeout=60.0)

                out = "".join(stdout_parts)
                err = ""
                with self._proc_lock:
                    self._proc = None

                tmp = self._config_temp
                if tmp and os.path.isfile(tmp):
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                self._config_temp = None

                if self._user_cancelled:
                    self.root.after(0, self._on_cancelled)
                    return

                # Worker writes JSON to INTEROPTIMUS_WORKER_RESULT (reliable in PyInstaller); stdout is fallback.
                text = ""
                try:
                    if os.path.isfile(worker_result_path):
                        text = (
                            Path(worker_result_path).read_text(encoding="utf-8", errors="replace") or ""
                        ).strip()
                except OSError:
                    text = ""
                if not text:
                    text = (out or "").strip()

                # Worker prints one JSON object to stdout even when ok=False (non-zero exit code).
                # Parse JSON before treating rc != 0 as a hard subprocess failure.
                payload: Optional[Dict[str, Any]] = None
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    for line in reversed(text.splitlines()):
                        line = line.strip()
                        if line.startswith("{"):
                            try:
                                payload = json.loads(line)
                                break
                            except json.JSONDecodeError:
                                continue

                if isinstance(payload, dict):
                    self.root.after(0, lambda p=payload: self._on_done(p))
                    return

                if rc != 0:
                    self.root.after(
                        0,
                        lambda: self._on_worker_fail((err or "").strip() or text or f"exit {rc}"),
                    )
                    return
                self.root.after(0, lambda: self._on_worker_fail(f"Invalid worker output:\n{text[:2000]}"))
            except Exception as e:
                with self._proc_lock:
                    self._proc = None
                self.root.after(0, lambda: self._on_error(e))
            finally:
                try:
                    if os.path.isfile(worker_result_path):
                        os.unlink(worker_result_path)
                except OSError:
                    pass
                self.root.after(0, lambda: setattr(self, "_viz_poll_active", False))

        threading.Thread(target=work, daemon=True).start()

    def _on_stop(self) -> None:
        self._user_cancelled = True
        with self._proc_lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
        except OSError:
            pass
        self._log_insert(self.t("cancelled"))

    def _on_cancelled(self) -> None:
        if self._run_btn:
            self._run_btn.config(state="normal")
        if self._stop_btn:
            self._stop_btn.config(state="disabled")
        if self._config_temp and os.path.isfile(self._config_temp):
            try:
                os.unlink(self._config_temp)
            except OSError:
                pass
            self._config_temp = None

    def _on_worker_fail(self, msg: str) -> None:
        if self._run_btn:
            self._run_btn.config(state="normal")
        if self._stop_btn:
            self._stop_btn.config(state="disabled")
        self._log_insert(f"\n{self.t('fail_worker')}: {msg}\n")
        messagebox.showerror(self.t("app_title"), f"{self.t('fail_worker')}\n{msg[:800]}")

    def _on_done(self, payload: Dict[str, Any]) -> None:
        if self._run_btn:
            self._run_btn.config(state="normal")
        if self._stop_btn:
            self._stop_btn.config(state="disabled")
        self._last_payload = payload
        if not payload.get("ok"):
            self._log_insert(f"\n失败: {payload.get('error', payload)}\n")
            if payload.get("traceback"):
                self._log_insert(payload["traceback"] + "\n")
            messagebox.showerror(self.t("app_title"), str(payload.get("error", "Unknown error")))
            return
        report = payload.get("report_text") or ""
        self._log_insert("\n--- OK ---\n")
        self._log_insert(report[:80000])
        if len(report) > 80000:
            self._log_insert("\n…\n")
        art = payload.get("artifacts") or {}
        wd = art.get("local_workdir", "")
        self._log_insert(f"\nworkdir: {wd}\n")
        messagebox.showinfo(self.t("done_title"), self.t("done_msg"))

    def _on_error(self, e: BaseException) -> None:
        if self._run_btn:
            self._run_btn.config(state="normal")
        if self._stop_btn:
            self._stop_btn.config(state="disabled")
        self._log_insert(f"\n异常: {e!r}\n")
        messagebox.showerror(self.t("app_title"), str(e))

    @staticmethod
    def _existing_file_path(p: Optional[str]) -> Optional[str]:
        """Return an absolute path only if the file exists (never resolve relative to process cwd)."""
        if not p or not isinstance(p, str):
            return None
        t = p.strip()
        if not t:
            return None
        try:
            expanded = os.path.expanduser(t)
            if not os.path.isabs(expanded):
                return None
            cand = os.path.normpath(expanded)
            if os.path.isfile(cand):
                return cand
            rp = os.path.realpath(cand)
            return rp if os.path.isfile(rp) else None
        except OSError:
            return None

    @staticmethod
    def _infer_run_dir_from_session(session_id: str) -> Optional[str]:
        """
        When ``local_workdir`` from the payload does not match on-disk state, locate outputs under
        ``sessions_root()/<session_id>/`` (same logic as ``local_workflow._run_dir_from_result``).
        """
        sid = (session_id or "").strip()
        if not sid:
            return None
        try:
            root = sessions_root() / sid
            if not root.is_dir():
                return None
            if (
                (root / "stereographic.jpg").is_file()
                or (root / "opt_results.pkl").is_file()
                or (root / "pairs_poscars.zip").is_file()
            ):
                return str(root.resolve())
            for sub in sorted(root.iterdir()):
                if sub.is_dir() and not sub.name.startswith("."):
                    if (
                        (sub / "stereographic.jpg").is_file()
                        or (sub / "opt_results.pkl").is_file()
                        or (sub / "pairs_poscars.zip").is_file()
                    ):
                        return str(sub.resolve())
        except OSError:
            return None
        return None

    def _resolve_artifact_workdir(self, art: Dict[str, Any], inner: Dict[str, Any]) -> Optional[str]:
        wd_raw = (art.get("local_workdir") or inner.get("local_workdir") or "").strip()
        if wd_raw:
            try:
                wdp = Path(wd_raw).expanduser().resolve(strict=False)
                if wdp.is_dir():
                    return str(wdp)
            except OSError:
                pass
            try:
                if os.path.isdir(wd_raw):
                    return os.path.abspath(os.path.expanduser(wd_raw))
            except OSError:
                pass
        sid = (self._last_payload.get("session_id") or "").strip()
        if sid:
            return self._infer_run_dir_from_session(sid)
        return None

    def _artifact_paths(self) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        if not self._last_payload or not self._last_payload.get("ok"):
            return None, None, None, None
        art = self._last_payload.get("artifacts") or {}
        inner = self._last_payload.get("result")
        if not isinstance(inner, dict):
            inner = {}

        wd_dir = self._resolve_artifact_workdir(art, inner)

        def pick(key: str, basename: str) -> Optional[str]:
            v = art.get(key)
            x = self._existing_file_path(v) if isinstance(v, str) else None
            if x:
                return x
            if wd_dir:
                return self._existing_file_path(str(Path(wd_dir) / basename))
            return None

        stereo = pick("stereographic_jpg", "stereographic.jpg")
        stereo_html = pick("stereographic_interactive_html", "stereographic_interactive.html")
        zpath = self._existing_file_path(art.get("poscars_zip_path")) if art.get("poscars_zip_path") else None
        if not zpath and wd_dir:
            zpath = self._existing_file_path(str(Path(wd_dir) / "pairs_poscars.zip"))

        if not wd_dir:
            if stereo:
                wd_dir = str(Path(stereo).parent)
            elif stereo_html:
                wd_dir = str(Path(stereo_html).parent)
            elif zpath:
                wd_dir = str(Path(zpath).parent)

        if not zpath and wd_dir:
            zpath = self._existing_file_path(str(Path(wd_dir) / "pairs_poscars.zip"))

        return wd_dir, stereo, stereo_html, zpath

    def _open_workdir(self) -> None:
        wd, _, _, _ = self._artifact_paths()
        if not wd or not os.path.isdir(wd):
            messagebox.showinfo(self.t("app_title"), self.t("folder_na"))
            return
        if sys.platform == "darwin":
            subprocess.run(["open", wd], check=False)
        elif sys.platform == "win32":
            subprocess.run(["explorer", wd], check=False)
        else:
            subprocess.run(["xdg-open", wd], check=False)

    def _open_stereo(self) -> None:
        _, s, _, _ = self._artifact_paths()
        if not s or not os.path.isfile(s):
            messagebox.showinfo(self.t("app_title"), self.t("file_na"))
            return
        if sys.platform == "darwin":
            subprocess.run(["open", s], check=False)
        elif sys.platform == "win32":
            os.startfile(s)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", s], check=False)

    def _open_stereo_html(self) -> None:
        _, _, h, _ = self._artifact_paths()
        if not h or not os.path.isfile(h):
            messagebox.showinfo(self.t("app_title"), self.t("file_na"))
            return
        if sys.platform == "darwin":
            subprocess.run(["open", h], check=False)
        elif sys.platform == "win32":
            os.startfile(h)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", h], check=False)

    def _open_zip(self) -> None:
        _, _, _, z = self._artifact_paths()
        if not z or not os.path.isfile(z):
            messagebox.showinfo(self.t("app_title"), self.t("file_na"))
            return
        if sys.platform == "darwin":
            subprocess.run(["open", z], check=False)
        elif sys.platform == "win32":
            os.startfile(z)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", z], check=False)


def run_gui() -> None:
    _silence_macos_tk_stderr_noise()
    root = tk.Tk()
    InterOptimusGui(root)
    root.mainloop()


if __name__ == "__main__":
    from InterOptimus.desktop_app.entry import main

    main()
