# Qibolab

[![codecov](https://codecov.io/gh/qiboteam/qibolab/branch/main/graph/badge.svg?token=11UENAPBPH)](https://codecov.io/gh/qiboteam/qibolab)
![PyPI - Version](https://img.shields.io/pypi/v/qibolab)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/qibolab)

Qibolab is the dedicated [Qibo](https://github.com/qiboteam/qibo) backend for
the automatic deployment of quantum circuits on quantum hardware.

Some of the key features of Qibolab are:

- Deploy Qibo models on quantum hardware easily.
- Create experimental drivers for custom lab setup.
- Support multiple heterogeneous platforms.
- Use calibration procedures from [Qibocal](https://github.com/qiboteam/qibocal).

## Documentation

[![docs](https://github.com/qiboteam/qibolab/actions/workflows/publish.yml/badge.svg)](https://qibo.science/qibolab/stable/)

The qibolab backend documentation is available at [https://qibo.science/qibolab/stable/](https://qibo.science/qibolab/stable/).

## Minimum working example

A simple example on how to connect to a platform and use it execute a pulse sequence:

```python
from qibolab import create_platform

# Define platform and load specific runcard
platform = create_platform("my_platform")

# Create a pulse sequence based on native gates of qubit 0
natives = platform.natives.single_qubit[0]
sequence = natives.RX() | natives.MZ()

# Connects to lab instruments using the details specified in the calibration settings.
platform.connect()

# Execute a pulse sequence
results = platform.execute([sequence], nshots=1000)

# Grab the acquired shots corresponding to
# the measurement using its pulse id.
# The ``PulseSequence`` structure is list[tuple[ChannelId, Pulse]]
# thererefore we need to index it appropriately
# to get the acquisition pulse
readout_id = sequence.acquisitions[0][1].id
print(results[readout_id])

# Disconnect from the instruments
platform.disconnect()
```

Arbitrary pulse sequences can also be created using the pulse API:

```python
from qibolab import (
    Acquisition,
    Delay,
    Gaussian,
    Pulse,
    PulseSequence,
    Readout,
    Rectangular,
)

# Crete some pulses
pulse = Pulse(
    amplitude=0.3,
    duration=40,
    relative_phase=0,
    envelope=Gaussian(rel_sigma=0.2),  # Gaussian shape with std = 0.2 * duration
)
delay = Delay(duration=40)
readout = Readout(
    acquisition=Acquisition(duration=2000),
    probe=Pulse(
        amplitude=0.9,
        duration=2000,
        envelope=Rectangular(),
        relative_phase=0,
    ),
)

# Add them to a PulseSequence
sequence = PulseSequence(
    [
        (1, pulse),  # pulse plays on channel 1
        (2, delay),  # delay and readout plays on channel 2
        (2, readout),
    ]
)
```

Here is another example on how to execute circuits:

```python
from qibo import gates, models, set_backend

# Create circuit and add native gates
c = models.Circuit(1)
c.add(gates.GPI2(0, phi=0.2))
c.add(gates.M(0))


# Simulate the circuit using numpy
set_backend("numpy")
result = c(nshots=1024)
print(result.probabilities())

# Execute the circuit on hardware
set_backend("qibolab", platform="my_platform")
result = c(nshots=1024)
print(result.probabilities())
```

## Citation policy

[![arXiv](https://img.shields.io/badge/arXiv-2308.06313-b31b1b.svg)](https://arxiv.org/abs/2308.06313)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10572987.svg)](https://doi.org/10.5281/zenodo.10572987)

If you use the package please refer to [the documentation](https://qibo.science/qibo/stable/appendix/citing-qibo.html#publications) for citation instructions.
