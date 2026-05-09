from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).resolve().parent
readme = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="InterOptimus",
    version="0.1.1",
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
    # Upper-bound caps protect users from unexpected breakage in fast-moving deps.
    # MLIP backends (torch, orb-models, sevenn, deepmd-kit, MatRIS) are NOT installed
    # here; use ``itom config --with-mlip-workers`` (see InterOptimus.deploy_jobflow_stack)
    # or install them manually into your worker / local conda env. The deploy script
    # pins torch to a version range whose PyPI default wheel is CUDA 12 (avoiding the
    # CUDA 13 default in torch 2.11+) so it works on offline / intranet-only clusters.
    install_requires=[
        "pymatgen>=2024.5,<2027",
        "interfacemaster",
        "scikit-optimize",
        "scikit-learn>=1.3,<2",
        "scipy>=1.11,<2",
        "pandas>=2.0,<3",
        "matplotlib>=3.7,<4",
        "numpy>=1.26,<2.3",
        "ase>=3.22,<4",
        "atomate2",
        "jobflow",
        "jobflow-remote",
        "qtoolkit",
        "adjustText",
        "tqdm>=4.65,<5",
        "mp-api",
        "pyyaml>=6.0,<7",
    ],
    extras_require={
        # Browser UI: interoptimus-web (see InterOptimus.web_app).
        "web": [
            "fastapi>=0.100,<1",
            "uvicorn[standard]>=0.22,<1",
            "python-multipart>=0.0.6",
            "jinja2>=3.1,<4",
            "plotly>=5.18,<7",
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
