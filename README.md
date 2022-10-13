# Qibolab

![Tests](https://github.com/qiboteam/qibolab/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/qiboteam/qibolab/branch/main/graph/badge.svg?token=11UENAPBPH)](https://codecov.io/gh/qiboteam/qibolab)
[![Documentation Status](https://readthedocs.org/projects/qibolab/badge/?version=latest)](https://qibolab.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/241307936.svg)](https://zenodo.org/badge/latestdoi/241307936)

Qibolab is the dedicated [Qibo](https://github.com/qiboteam/qibo) backend for
the automatic deployment of quantum circuits on quantum hardware.

Some of the key features of Qibolab are:

* Deploy Qibo models on quantum hardware easily.
* Create custom experimental drivers for custom lab setup.
* Support multiple heterogeneous platforms.
* Use existing calibration procedures for experimentalists.

## Documentation

The qibolab backend documentation is available at https://qibolab.readthedocs.io.

## Minimum working example

A simple example on how to connect to the TIIq platform and use it execute a pulse sequence:

```python
from qibolab import Platform
from qibolab.paths import qibolab_folder
from qibolab.pulses import Pulse, ReadoutPulse, PulseSequence

# Define PulseSequence
sequence = PulseSequence()
# Add some pulses to the pulse sequence
sequence.add(Pulse(start=0,
                   amplitude=0.3,
                   duration=4000,
                   frequency=200_000_000,
                   relative_phase=0,
                   shape='Gaussian(5)', # Gaussian shape with std = duration / 5
                   channel=1))

sequence.add(ReadoutPulse(start=4004,
                          amplitude=0.9,
                          duration=2000,
                          frequency=20_000_000,
                          relative_phase=0,
                          shape='Rectangular',
                          channel=2))

# Define platform and load specific runcard
runcard = qibolab_folder / 'runcards' / 'tii1q.yml'
platform = Platform("tii1q", runcard)

# Connects to lab instruments using the details specified in the calibration settings.
platform.connect()
# Configures instruments using the loaded calibration settings.
platform.setup()
# Turns on the local oscillators
platform.start()
# Executes a pulse sequence.
results = platform.execute_pulse_sequence(sequence, nshots=3000)
print(f"results (amplitude, phase, i, q): {results}")
# Turn off lab instruments
platform.stop()
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
qibo.set_backend("qibolab", platform="tii1q")

for _ in range(5):
    result = c(nshots=1024)
    print(result.probabilities())
```

## Citation policy

If you use the package please cite the following references:
- https://arxiv.org/abs/2009.01845
- https://doi.org/10.5281/zenodo.3997194
- DOI paper and zenodo
