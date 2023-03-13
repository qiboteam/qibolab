# Installation script for python
import os
import re

from setuptools import find_packages, setup

PACKAGE = "qibolab"


# Returns the version
def get_version():
    """Gets the version from the package's __init__ file
    if there is some problem, let it happily fail"""
    VERSIONFILE = os.path.join("src", PACKAGE, "__init__.py")
    initfile_lines = open(VERSIONFILE).readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)


# load long description from README
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name=PACKAGE,
    version=get_version(),
    description="Quantum hardware module and drivers for Qibo",
    author="The Qibo team",
    author_email="",
    url="https://github.com/qiboteam/qibolab",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["runcards/*.yml", "tests/*.yml"]},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=[
        "qibo>=0.1.8",
        "networkx",
        "more-itertools",
    ],
    extras_require={
        "docs": [
            "sphinx",
            "furo",
            "recommonmark",
            "sphinxcontrib-bibtex",
            "sphinx_markdown_tables",
            "nbsphinx",
            "IPython",
            "sphinx_copybutton",
            "furo",
        ],
        "tests": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "pytest-env>=0.8.1",
        ],
        "analysis": [
            "pylint>=2.16.0",
        ],
        # TII system dependencies
        "tiiq": [
            "qblox-instruments==0.7.0",
            "qcodes",
            "pyvisa-py==0.5.3",
            "qm-qua>=1.0.1",
            "qualang-tools>=0.13.1",
        ],
    },
    python_requires=">=3.8.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
