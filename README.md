# Qibolab

[![codecov](https://codecov.io/gh/qiboteam/qibolab/branch/main/graph/badge.svg?token=11UENAPBPH)](https://codecov.io/gh/qiboteam/qibolab)
![PyPI - Version](https://img.shields.io/pypi/v/qibolab)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/qibolab)

Qibolab is the dedicated [Qibo](https://github.com/qiboteam/qibo) backend for
the automatic deployment of quantum circuits on quantum hardware.

Some of the key features of Qibolab are:

- Deploy Qibo models on quantum hardware easily.
- Create custom experimental drivers for custom lab setup.
- Support multiple heterogeneous platforms.
- Use existing calibration procedures for experimentalists.

## Documentation

[![docs](https://github.com/qiboteam/qibolab/actions/workflows/publish.yml/badge.svg)](https://qibo.science/qibolab/stable/)

The qibolab backend documentation is available at [https://qibo.science/qibolab/stable/](https://qibo.science/qibolab/stable/).

## Minimum working example

A simple example on how to connect to a platform and use it execute a pulse sequence:

```python
from qibolab import create_platform, ExecutionParameters
from qibolab.pulses import Pulse, Delay, PulseType

# Define PulseSequence
sequence = PulseSequence()
# Add some pulses to the pulse sequence
sequence.append(
    Pulse(
        amplitude=0.3,
        duration=4000,
        frequency=200_000_000,
        relative_phase=0,
        shape="Gaussian(5)",  # Gaussian shape with std = duration / 5
        type=PulseType.DRIVE,
        channel=1,
    )
)
sequence.append(
    Delay(
        duration=4000,
        channel=2,
    )
)
sequence.append(
    ReadoutPulse(
        amplitude=0.9,
        duration=2000,
        frequency=20_000_000,
        relative_phase=0,
        shape="Rectangular",
        type=PulseType.READOUT,
        channel=2,
    )
)

# Define platform and load specific runcard
platform = create_platform("my_platform")

# Connects to lab instruments using the details specified in the calibration settings.
platform.connect()

# Execute a pulse sequence
options = ExecutionParameters(nshots=1000)
results = platform.execute([sequence], options)

# Print the acquired shots
print(results.samples)

# Disconnect from the instruments
platform.disconnect()
```

Here is another example on how to execute circuits:

```python
import qibo
from qibo import gates, models


# Create circuit and add gates
c = models.Circuit(1)
c.add(gates.H(0))
c.add(gates.RX(0, theta=0.2))
c.add(gates.X(0))
c.add(gates.M(0))


# Simulate the circuit using numpy
qibo.set_backend("numpy")
for _ in range(5):
    result = c(nshots=1024)
    print(result.probabilities())

# Execute the circuit on hardware
qibo.set_backend("qibolab", platform="my_platform")
for _ in range(5):
    result = c(nshots=1024)
    print(result.probabilities())
```

## Citation policy

[![arXiv](https://img.shields.io/badge/arXiv-2308.06313-b31b1b.svg)](https://arxiv.org/abs/2308.06313)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10572987.svg)](https://doi.org/10.5281/zenodo.10572987)

If you use the package please refer to [the documentation](https://qibo.science/qibo/stable/appendix/citing-qibo.html#publications) for citation instructions.
