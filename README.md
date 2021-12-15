# qibolab

![Tests](https://github.com/qiboteam/qibolab/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/qiboteam/qibolab/branch/main/graph/badge.svg?token=11UENAPBPH)](https://codecov.io/gh/qiboteam/qibolab)
ZENODO_BADGE

This package provides the quantum hardware control implementation for multi-platform.

## Documentation

The qibolab backend documentation is available at [qibolab.readthedocs.io](http://34.240.99.72/qibolab/) (username:qiboteam, pass:qkdsimulator).

## Minimum working example

A simple example on how to import the TIIq platform and use it execute a pulse sequence:

```python
from qibolab import platform, pulses # automatically connects to the platform
from qibolab.pulse_shapes import Rectangular, Gaussian


# define pulse sequence
sequence = pulses.PulseSequence()
sequence.add(pulses.Pulse(start=0,
                          frequency=200000000.0,
                          amplitude=0.3,
                          duration=60,
                          phase=0,
                          shape=Gaussian(60 / 5))) # Gaussian shape with std = duration / 5
sequence.add(pulses.ReadoutPulse(start=70,
                                 frequency=20000000.0,
                                 amplitude=0.5,
                                 duration=3000,
                                 phase=0,
                                 shape=Rectangular()))

# turn on instruments
platform.start()
# execute sequence and acquire results
results = platform.execute(sequence)
# turn off instruments
platform.stop()
```

## Citation policy

If you use the package please cite the following references:
- https://arxiv.org/abs/2009.01845
- https://doi.org/10.5281/zenodo.3997194
- DOI paper and zenodo
