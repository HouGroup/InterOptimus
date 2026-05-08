from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).resolve().parent
readme = (here / "README.md").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Version pinning policy
# ---------------------------------------------------------------------------
# We pin **upper bounds** on the science stack so a broken upstream major
# release cannot silently break a fresh `pip install`. Lower bounds reflect
# the oldest API our code is verified against.
#
# CUDA 12 (torch): starting with **PyTorch 2.11**, the default PyPI wheel
# (`pip install torch`) ships **CUDA 13.0** binaries. We want CUDA 12, so the
# `mlip-cu12` extras pins ``torch < 2.11`` — this keeps PyPI's default wheel on
# CUDA 12.x and also matches our checkpoint files. Users who need a specific
# CUDA 12 minor (12.6 / 12.8) can install from the PyTorch index directly,
# e.g. ``pip install --extra-index-url https://download.pytorch.org/whl/cu128
# 'torch>=2.4,<2.11'``. CUDA 13 is intentionally NOT supported here because
# Maxwell / Pascal GPUs are dropped and several of our MLIP backends still
# bundle CUDA 12 kernels.
# ---------------------------------------------------------------------------

CORE_REQUIRES = [
    # Materials / structure stack
    "pymatgen>=2024.5.1,<2026",
    "interfacemaster>=2.0,<3",
    # Optimization / scientific computing
    "scikit-optimize>=0.9.0,<1",
    "scikit-learn>=1.3,<2",
    "scipy>=1.10,<2",
    "pandas>=2.0,<3",
    "matplotlib>=3.7,<4",
    # NumPy 2.x is supported; cap below NumPy 3 just in case.
    "numpy>=1.24,<3",
    # ASE 3.x; 4.x is not yet shipped.
    "ase>=3.22,<4",
    # Jobflow stack — pre-1.0, pin under the next minor that breaks API.
    "atomate2>=0.0.16,<0.1",
    "jobflow>=0.1.18,<0.3",
    "jobflow-remote>=0.1.5,<0.3",
    "qtoolkit>=0.1.4,<1",
    # Misc
    "adjustText>=0.8,<2",
    "tqdm>=4.60,<5",
    "mp-api>=0.41,<1",
    "pyyaml>=6.0,<7",
]

WEB_REQUIRES = [
    "fastapi>=0.100,<1",
    "uvicorn[standard]>=0.22,<1",
    "python-multipart>=0.0.6,<1",
    "jinja2>=3.1,<4",
    "plotly>=5.18,<7",
]

# --- MLIP backends -----------------------------------------------------------
# These pull torch (CUDA 12 path); each backend is gated to its own extras so a
# minimal install does not download a >1 GB torch wheel. Calculator imports in
# ``InterOptimus.mlip`` happen lazily — the Python package is only required on
# the worker that actually runs that calculator.
#
# Torch upper bound is the **single source of CUDA 12 enforcement**. Bumping it
# above 2.11 implies CUDA 13 wheels — do not change without updating this note.
TORCH_CU12 = "torch>=2.4,<2.11"

MLIP_CU12_REQUIRES = [
    TORCH_CU12,
]

# Optional per-backend bundles. Activate explicitly because they are heavy and
# their own pins can conflict (e.g. orb-models tracks newer ASE/numpy than DPA).
ORB_REQUIRES = [TORCH_CU12, "orb-models>=0.5,<1"]
SEVENN_REQUIRES = [TORCH_CU12, "sevenn>=0.10,<1"]
MATRIS_REQUIRES = [TORCH_CU12, "matris>=0.5,<1"]
DEEPMD_REQUIRES = [TORCH_CU12, "deepmd-kit>=3.0,<4"]

setup(
    name="InterOptimus",
    version="0.1.0",
    author="Yaoshu Xie",
    author_email="jasonxie@sz.tsinghua.edu.cn",
    description="High throughput simulation for crystalline interfaces",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/HouGroup/InterOptimus/",
    project_urls={
        "Source": "https://github.com/HouGroup/InterOptimus/",
        "Documentation": "https://github.com/HouGroup/InterOptimus/blob/HEAD/docs/GETTING_STARTED.md",
    },
    license="MIT",
    keywords="materials science, interfaces, MLIP, VASP, jobflow, pymatgen",
    packages=find_packages(exclude=("tests", "test", "new_test")),
    include_package_data=True,
    package_data={
        "InterOptimus.web_app": ["templates/*.html"],
        "InterOptimus.agents": ["*.json"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10,<3.13",
    install_requires=CORE_REQUIRES,
    extras_require={
        # Browser UI: interoptimus-web (see InterOptimus.web_app).
        "web": WEB_REQUIRES,
        # Plain torch on CUDA 12 — the minimum required for any MLIP backend.
        # ``pip install -e '.[mlip-cu12]'`` (or pass via ``--extra-index-url`` for
        # a specific CUDA 12 minor, see the policy comment above).
        "mlip-cu12": MLIP_CU12_REQUIRES,
        # Per-backend bundles (each pulls torch CUDA 12).
        "mlip-orb": ORB_REQUIRES,
        "mlip-sevenn": SEVENN_REQUIRES,
        "mlip-matris": MATRIS_REQUIRES,
        "mlip-deepmd": DEEPMD_REQUIRES,
        # Convenience meta-extra: torch CUDA 12 + every supported backend.
        "mlip-all-cu12": MLIP_CU12_REQUIRES + [
            "orb-models>=0.5,<1",
            "sevenn>=0.10,<1",
            "matris>=0.5,<1",
            "deepmd-kit>=3.0,<4",
        ],
    },
    entry_points={
        "console_scripts": [
            "itom=InterOptimus.cli:main",
            "interoptimus-env=InterOptimus.agents.server_env:main",
            "interoptimus-simple=InterOptimus.agents.simple_iomaker:main",
            "interoptimus-web=InterOptimus.web_app.cli:main",
        ],
    },
)
