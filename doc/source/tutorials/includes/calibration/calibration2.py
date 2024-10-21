import matplotlib.pyplot as plt

from qibolab import create_platform
from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.pulses import PulseSequence

# allocate platform
platform = create_platform("dummy")

# create pulse sequence 1 and add pulses
one_sequence = PulseSequence()
drive_pulse = platform.create_RX_pulse(qubit=0, start=0)
readout_pulse1 = platform.create_MZ_pulse(qubit=0, start=drive_pulse.finish)
one_sequence.add(drive_pulse)
one_sequence.add(readout_pulse1)

# create pulse sequence 2 and add pulses
zero_sequence = PulseSequence()
readout_pulse2 = platform.create_MZ_pulse(qubit=0, start=0)
zero_sequence.add(readout_pulse2)

options = ExecutionParameters(
    nshots=1000,
    relaxation_time=50_000,
    averaging_mode=AveragingMode.SINGLESHOT,
    acquisition_type=AcquisitionType.INTEGRATION,
)

results_one = platform.execute_pulse_sequence(one_sequence, options)
results_zero = platform.execute_pulse_sequence(zero_sequence, options)

plt.title("Single shot classification")
plt.xlabel("I [a.u.]")
plt.ylabel("Q [a.u.]")
plt.scatter(
    results_one[readout_pulse1.serial].voltage_i,
    results_one[readout_pulse1.serial].voltage_q,
    label="One state",
)
plt.scatter(
    results_zero[readout_pulse2.serial].voltage_i,
    results_zero[readout_pulse2.serial].voltage_q,
    label="Zero state",
)
plt.show()
