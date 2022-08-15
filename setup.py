# Installation script for python
from setuptools import setup, find_packages
import os
import re

PACKAGE = "qibolab"


# Returns the version
def get_version():
    """ Gets the version from the package's __init__ file
    if there is some problem, let it happily fail """
    VERSIONFILE = os.path.join("src", PACKAGE, "__init__.py")
    initfile_lines = open(VERSIONFILE, "rt").readlines()
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
    description="Quantum hardware backend for Qibo",
    author="The Qibo team",
    author_email="",
    url="https://github.com/qiboteam/qibolab",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["runcards/*.yml", "tests/*.yml", "*.txt"]},
    entry_points={'console_scripts': ['calibrate-qubits = qibolab.calibration.calibrate_qubits:main']},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=[
        "qibo>=0.1.8rc0",
        "qcodes",   # 0.34.1
        "lmfit",    # 1.0.3
    ],
    extras_require={
        "docs": [
            "sphinx",
            "sphinx_rtd_theme",
            "recommonmark",
            "sphinxcontrib-bibtex",
            "sphinx_markdown_tables",
            "nbsphinx",
            "IPython"],
        # TII system dependencies
        "tiiq": [
            "qblox-instruments=0.6.1",    # 0.6.1
            "quantify-core",        # 0.6.0
            "dash",                 # Live plotting
        ]
    },
    python_requires=">=3.6.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
)
