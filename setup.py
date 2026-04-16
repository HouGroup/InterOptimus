from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).resolve().parent
readme = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="InterOptimus",
    version="0.0.4",
    author="Yaoshu Xie",
    author_email="jasonxie@sz.tsinghua.edu.cn",
    description="High throughput simulation for crystalline interfaces",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/HouGroup/InterOptimus/",
    license="MIT",
    packages=find_packages(exclude=("tests", "test", "new_test")),
    include_package_data=True,
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
        "dscribe",
        "scikit-optimize",
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
        "orb-models",
        "sevenn",
        "deepmd-kit",
        "torch",
    ],
    entry_points={
        "console_scripts": [
            "interoptimus-env=InterOptimus.agents.server_env:main",
            "interoptimus-simple=InterOptimus.agents.simple_iomaker:main",
        ],
    },
)
