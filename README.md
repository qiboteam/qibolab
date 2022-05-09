# Qibolab

![Tests](https://github.com/qiboteam/qibolab/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/qiboteam/qibolab/branch/main/graph/badge.svg?token=11UENAPBPH)](https://codecov.io/gh/qiboteam/qibolab)
ZENODO_BADGE

Qibolab is the dedicated [Qibo](https://github.com/qiboteam/qibo) backend for
the automatic deployment of quantum circuits on quantum hardware.

Some of the key features of Qibolab are:

* Deploy Qibo models on quantum hardware easily.
* Create custom experimental drivers for custom lab setup.
* Support multiple heterogeneous platforms.
* Use existing calibration procedures for experimentalists.

## Documentation

The qibolab backend documentation is available at [qibolab.readthedocs.io](http://34.240.99.72/qibolab/) (username:qiboteam, pass:qkdsimulator).

## Minimum working example

A simple example on how to connect to the TIIq platform and use it execute a pulse sequence:

```python
from qibolab import Platform
from qibolab.pulses import Pulse, ReadoutPulse
from qibolab.circuit import PulseSequence
from qibolab.pulse_shapes import Rectangular, Gaussian

# Define PulseSequence
sequence = PulseSequence()

# Add some pulses to the pulse sequence
sequence.add(Pulse(start=0,
                   frequency=200000000.0,
                   amplitude=0.3,
                   duration=60,
                   phase=0,
                   shape=Gaussian(5))) # Gaussian shape with std = duration / 5

sequence.add(ReadoutPulse(start=70,
                          frequency=20000000.0,
                          amplitude=0.5,
                          duration=3000,
                          phase=0,
                          shape=Rectangular()))

# Define platform and load specific runcard
platform = Platform("tiiq")
# Connects to lab instruments using the details specified in the calibration settings.
platform.connect()
# Configures instruments using the loaded calibration settings.
platform.setup()
# Turns on the local oscillators
platform.start()
# Executes a pulse sequence.
results = platform.execute(sequence, nshots=10)
# Turn off lab instruments
platform.stop()
# Disconnect from the instruments
platform.disconnect()
```

## Citation policy

If you use the package please cite the following references:
- https://arxiv.org/abs/2009.01845
- https://doi.org/10.5281/zenodo.3997194
- DOI paper and zenodo
