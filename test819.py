import numpy as np
from qibo.backends import GlobalBackend

from qibolab import Platform
from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper, SweeperType

GlobalBackend.set_backend("qibolab", "spinq10q")
platform: Platform = GlobalBackend().platform

sequence = PulseSequence()
ro_pulses = {}
qubit_frequency = {}
bare_resonator_frequency = {}

qid = 0
qubit = platform.qubits[qid]
qubit_frequency[qid] = qubit.drive_frequency
bare_resonator_frequency[qid] = qubit.bare_resonator_frequency

ro_pulses[qid] = platform.create_qubit_readout_pulse(qid, start=0)
sequence.add(ro_pulses[qid])

freq_width = 2e7
freq_step = 2e5
bias_width = 0.8
bias_step = 0.01
nshots = 1024
relaxation_time = 5000

# define the parameters to sweep and their range:
delta_frequency_range = np.arange(-freq_width // 2, freq_width // 2, freq_step)
freq_sweeper = Sweeper(
    Parameter.frequency,
    delta_frequency_range,
    [ro_pulses[qid]],
    type=SweeperType.OFFSET,
)

delta_bias_range = np.arange(-bias_width / 2, bias_width / 2, bias_step)
bias_sweepers = [
    Sweeper(Parameter.bias, delta_bias_range, qubits=[qubit], type=SweeperType.OFFSET)
]

options = ExecutionParameters(
    nshots=nshots,
    relaxation_time=relaxation_time,
    acquisition_type=AcquisitionType.INTEGRATION,
    averaging_mode=AveragingMode.CYCLIC,
)
for bias_sweeper in bias_sweepers:
    __import__("pdb").set_trace()
    results = platform.sweep(sequence, options, bias_sweeper, freq_sweeper)
