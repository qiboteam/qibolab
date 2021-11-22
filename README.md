# qibolab

![Tests](https://github.com/qiboteam/qibolab/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/qiboteam/qibolab/branch/main/graph/badge.svg?token=11UENAPBPH)](https://codecov.io/gh/qiboteam/qibolab)
ZENODO_BADGE

This package provides the quantum hardware control implementation for multi-platform.

## Documentation

The qibolab backend documentation is available at [qibolab.readthedocs.io](http://34.240.99.72/qibolab/) (username:qiboteam, pass:qkdsimulator).

## Installation

```
conda create --name Quantum python=3.8
conda activate Quantum
conda install -c conda-forge jupyterlab
python -m ipykernel install --user --name=Quantum  --display-name="Quantum Env"

pip install qblox-instruments==0.4.0
pip install lmfit

pip install quantify-core
pip install qibo


pip install pyvisa-py
```



```
python setup.py develop easy_install qibolab[tiiq]
```



## Citation policy

If you use the package please cite the following references:
- https://arxiv.org/abs/2009.01845
- https://doi.org/10.5281/zenodo.3997194
- DOI paper and zenodo
