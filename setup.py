from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).resolve().parent
readme = (here / "README.md").read_text(encoding="utf-8")

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
    python_requires=">=3.10",
    install_requires=[
        "pymatgen",
        "interfacemaster",
        "scikit-optimize",
        "scikit-learn",
        "scipy",
        "pandas",
        "matplotlib",
        "numpy",
        "ase",
        "atomate2",
        "jobflow",
        "jobflow-remote",
        "qtoolkit",
        "adjustText",
        "tqdm",
        "mp-api",
        "pyyaml>=6.0",
    ],
    extras_require={
        # MLIP backends (torch, orb-models, …) are not bundled here—use `itom config --with-mlip-workers`
        # or install those packages manually into your worker / local conda env.
        # Browser UI: interoptimus-web (see InterOptimus.web_app).
        "web": [
            "fastapi>=0.100",
            "uvicorn[standard]>=0.22",
            "python-multipart>=0.0.6",
            "jinja2>=3.1",
            "plotly>=5.18",
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
