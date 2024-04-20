import numpy as np

from qibolab import create_platform
from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper, SweeperType

# allocate platform
platform = create_platform("dummy")

# create pulse sequence and add pulse
sequence = PulseSequence()
readout_pulse = platform.create_MZ_pulse(qubit=0, start=0)
sequence.add(readout_pulse)

# allocate frequency sweeper
sweeper = Sweeper(
    parameter=Parameter.frequency,
    values=np.arange(-2e8, +2e8, 1e6),
    pulses=[readout_pulse],
    type=SweeperType.OFFSET,
)

options = ExecutionParameters(
    nshots=1000,
    relaxation_time=50,
    averaging_mode=AveragingMode.CYCLIC,
    acquisition_type=AcquisitionType.INTEGRATION,
)

results = platform.sweep(sequence, options, sweeper)

import matplotlib.pyplot as plt

amplitudes = results[readout_pulse.serial].magnitude
frequencies = np.arange(-2e8, +2e8, 1e6) + readout_pulse.frequency

plt.title("Resonator Spectroscopy")
plt.xlabel("Frequencies [Hz]")
plt.ylabel("Amplitudes [a.u.]")

plt.plot(frequencies, amplitudes)
plt.show()
