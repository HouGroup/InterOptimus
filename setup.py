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
    package_data={},
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
        "torch",
        "eqnorm @ git+https://github.com/yzchen08/eqnorm.git",
    ],
    extras_require={
        "web": [
            "fastapi>=0.100",
            "uvicorn[standard]>=0.22",
            "python-multipart>=0.0.6",
        ],
        "desktop": [
            "pyinstaller>=6.0",
            "tkinterweb>=4.0",
        ],
        # Eqnorm + PyG on macOS: PyTorch 2.11+ 常与 PyG 预编译轮子的索引/ABI 踩坑；2.6.x 更稳。
        # 用法: pip install -e ".[eqnorm-torch26]" 或见 desktop/install_eqnorm_torch26.sh
        "eqnorm-torch26": [
            "torch>=2.6.0,<2.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "interoptimus-env=InterOptimus.agents.server_env:main",
            "interoptimus-simple=InterOptimus.agents.simple_iomaker:main",
            "interoptimus-desktop=InterOptimus.desktop_app.entry:main",
        ],
    },
)
