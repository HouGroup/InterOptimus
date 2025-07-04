from setuptools import setup, find_packages

setup(
    name="InterOptimus",
    version="0.0.4",
    author="Yaoshu Xie",
    author_email="jasonxie@sz.tsinghua.edu.cn",
    description="High througput simulation making crystalline interfaces",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HouGroup/InterOptimus/",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pymatgen",
        "interfacemaster",
        "Dscribe",
        "scikit-optimize",
        "matplotlib",
        "atomate",
        "fireworks",
        "adjustText",
        "ipywidgets",
        "tqdm",
        "mlipdockers",
        "mp-api",
        "orb-models",
        "sevenn",
    ],
    include_package_data=True,
    package_data={
        'InterOptimus': ['/data/orb-v3-conservative-20-omat-20250404.ckpt',
            '/data/checkpoint_sevennet_mf_ompa.pth']
            },
)
