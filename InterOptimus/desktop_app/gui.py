"""
Native Tkinter GUI for the MatRIS ``simple_iomaker`` workflow (no browser / no FastAPI).
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
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# i18n
# ---------------------------------------------------------------------------

STRINGS: Dict[str, Dict[str, str]] = {
    "zh": {
        "app_title": "InterOptimus · MatRIS",
        "lang": "语言 Language",
        "cif_files": "CIF 文件",
        "film_cif": "薄膜 film.cif",
        "substrate_cif": "基底 substrate.cif",
        "browse": "浏览…",
        "tab_basic": "基本",
        "tab_lm": "点阵匹配",
        "tab_structure": "结构",
        "tab_opt": "优化 (MatRIS)",
        "workflow_name": "工作流名称",
        "cost_preset": "成本预设",
        "double_interface": "双界面模型 (double_interface)",
        "execution": "执行方式",
        "exec_local": "本机 local",
        "exec_server": "集群 server",
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
    },
    "en": {
        "app_title": "InterOptimus · MatRIS",
        "lang": "Language 语言",
        "cif_files": "CIF files",
        "film_cif": "Film film.cif",
        "substrate_cif": "Substrate substrate.cif",
        "browse": "Browse…",
        "tab_basic": "Basic",
        "tab_lm": "Lattice match",
        "tab_structure": "Structure",
        "tab_opt": "Optimization (MatRIS)",
        "workflow_name": "Workflow name",
        "cost_preset": "Cost preset",
        "double_interface": "Double-interface model",
        "execution": "Execution",
        "exec_local": "Local",
        "exec_server": "Server / cluster",
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
        self._workflow = tk.StringVar(value="IO_web_matris")
        self._cost = tk.StringVar(value="medium")
        self._double_if = tk.BooleanVar(value=False)
        self._exec = tk.StringVar(value="local")

        self._lm_area = tk.StringVar(value="60")
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
        # Light theme palette
        self._bg = "#f4f6fa"
        self._panel = "#ffffff"
        self._accent = "#2563eb"
        self._muted = "#64748b"
        self.root.configure(bg=self._bg)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Main.TFrame", background=self._bg)
        style.configure("Card.TLabelframe", background=self._panel, relief="flat", borderwidth=1)
        style.configure("Card.TLabelframe.Label", background=self._panel, foreground=self._muted, font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background=self._bg, foreground="#1e293b")
        style.configure("Card.TLabel", background=self._panel, foreground="#334155")
        style.configure("TEntry", fieldbackground="#ffffff")
        style.configure("Accent.TButton", foreground="#ffffff")
        style.map(
            "Accent.TButton",
            background=[("active", "#1d4ed8"), ("!disabled", self._accent)],
        )
        style.configure("Stop.TButton", foreground="#ffffff")
        style.map("Stop.TButton", background=[("active", "#b91c1c"), ("!disabled", "#dc2626")])

        # Title bar
        try:
            self._title_font = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        except tk.TclError:
            self._title_font = tkfont.Font(size=16, weight="bold")

    def _build(self) -> None:
        pad = {"padx": 12, "pady": 6}
        outer = ttk.Frame(self.root, style="Main.TFrame")
        outer.pack(fill="both", expand=True)

        # Header
        hdr = ttk.Frame(outer, style="Main.TFrame")
        hdr.pack(fill="x", **pad)
        title = tk.Label(
            hdr,
            text=self.t("app_title"),
            font=self._title_font,
            bg=self._bg,
            fg="#0f172a",
        )
        title.pack(side="left")
        self._i18n_widgets.append((title, "app_title", lambda w, s: w.config(text=s)))

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

        # CIF card
        f_top = ttk.LabelFrame(outer, text=self.t("cif_files"), style="Card.TLabelframe")
        f_top.pack(fill="x", **pad)
        self._i18n_widgets.append((f_top, "cif_files", lambda w, s: w.config(text=s)))

        r1 = ttk.Frame(f_top)
        r1.pack(fill="x", padx=10, pady=6)
        lb1 = ttk.Label(r1, text=self.t("film_cif"), style="Card.TLabel", width=18)
        lb1.pack(side="left")
        self._i18n_widgets.append((lb1, "film_cif", lambda w, s: w.config(text=s)))
        ttk.Entry(r1, textvariable=self._film, width=50).pack(side="left", fill="x", expand=True, padx=4)
        b1 = ttk.Button(r1, text=self.t("browse"), command=self._browse_film)
        b1.pack(side="left")
        self._i18n_widgets.append((b1, "browse", lambda w, s: w.config(text=s)))

        r2 = ttk.Frame(f_top)
        r2.pack(fill="x", padx=10, pady=(0, 10))
        lb2 = ttk.Label(r2, text=self.t("substrate_cif"), style="Card.TLabel", width=18)
        lb2.pack(side="left")
        self._i18n_widgets.append((lb2, "substrate_cif", lambda w, s: w.config(text=s)))
        ttk.Entry(r2, textvariable=self._sub, width=50).pack(side="left", fill="x", expand=True, padx=4)
        b2 = ttk.Button(r2, text=self.t("browse"), command=self._browse_sub)
        b2.pack(side="left")
        self._i18n_widgets.append((b2, "browse", lambda w, s: w.config(text=s)))

        # Notebook
        self._notebook = ttk.Notebook(outer)
        self._notebook.pack(fill="both", expand=True, padx=12, pady=(0, 6))

        t_basic = ttk.Frame(self._notebook)
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

        lb_ex = ttk.Label(t_basic, text=self.t("execution"))
        lb_ex.grid(row=row, column=0, sticky="nw", padx=10, pady=4)
        self._i18n_widgets.append((lb_ex, "execution", lambda w, s: w.config(text=s)))
        rf = ttk.Frame(t_basic)
        rf.grid(row=row, column=1, sticky="w", padx=10, pady=4)
        rb_loc = ttk.Radiobutton(rf, text=self.t("exec_local"), variable=self._exec, value="local")
        rb_loc.pack(anchor="w")
        rb_srv = ttk.Radiobutton(rf, text=self.t("exec_server"), variable=self._exec, value="server")
        rb_srv.pack(anchor="w")
        self._i18n_widgets.append((rb_loc, "exec_local", lambda w, s: w.config(text=s)))
        self._i18n_widgets.append((rb_srv, "exec_server", lambda w, s: w.config(text=s)))

        t_lm = ttk.Frame(self._notebook)
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

        t_st = ttk.Frame(self._notebook)
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

        t_op = ttk.Frame(self._notebook)
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

        f_act = ttk.Frame(outer, style="Main.TFrame")
        f_act.pack(fill="x", padx=12, pady=8)
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
            b = ttk.Button(f_act, text=self.t(key), command=cmd)
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
            font=("Menlo", 10) if sys.platform == "darwin" else ("Consolas", 10),
            bg="#f8fafc",
            fg="#0f172a",
            insertbackground="#0f172a",
            relief="flat",
            padx=8,
            pady=8,
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
        tabs = ["tab_basic", "tab_lm", "tab_structure", "tab_opt"]
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
            "cost_preset": self._cost.get(),
            "double_interface": _bool_to_str(self._double_if.get()),
            "execution": self._exec.get(),
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

        form = self._collect_form()
        fd, path = tempfile.mkstemp(suffix=".json", prefix="io_matris_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"film": fp, "sub": sp, "form": form}, f, ensure_ascii=False)
        except OSError as e:
            messagebox.showerror(self.t("app_title"), str(e))
            return

        self._config_temp = path
        self._user_cancelled = False
        self._log_insert(self.t("running"))
        if self._run_btn:
            self._run_btn.config(state="disabled")
        if self._stop_btn:
            self._stop_btn.config(state="normal")

        cmd = [sys.executable, "-m", "InterOptimus.desktop_app.worker", path]

        def work() -> None:
            proc: Optional[subprocess.Popen[str]] = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                with self._proc_lock:
                    self._proc = proc
                out, err = proc.communicate()
                rc = proc.returncode
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
                if rc != 0:
                    self.root.after(
                        0,
                        lambda: self._on_worker_fail((err or "").strip() or (out or "").strip() or f"exit {rc}"),
                    )
                    return
                try:
                    payload = json.loads(out or "{}")
                except json.JSONDecodeError as e:
                    self.root.after(0, lambda: self._on_worker_fail(f"JSON: {e}\n{(out or '')[:2000]}"))
                    return
                self.root.after(0, lambda p=payload: self._on_done(p))
            except Exception as e:
                with self._proc_lock:
                    self._proc = None
                self.root.after(0, lambda: self._on_error(e))

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

    def _artifact_paths(self) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        if not self._last_payload or not self._last_payload.get("ok"):
            return None, None, None, None
        art = self._last_payload.get("artifacts") or {}
        wd = art.get("local_workdir") or ""
        z = art.get("poscars_zip_path")
        stereo = os.path.join(wd, "stereographic.jpg") if wd else None
        stereo_html = os.path.join(wd, "stereographic_interactive.html") if wd else None
        if stereo and not os.path.isfile(stereo):
            stereo = None
        if stereo_html and not os.path.isfile(stereo_html):
            stereo_html = None
        if z and not os.path.isfile(z):
            z = None
        return wd, stereo, stereo_html, z

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
    root = tk.Tk()
    InterOptimusGui(root)
    root.mainloop()


if __name__ == "__main__":
    from InterOptimus.desktop_app.entry import main

    main()
