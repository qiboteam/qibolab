import numpy as np
from qibo.backends import GlobalBackend

from qibolab import Platform
from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.instruments.qblox.controller import QbloxController
from qibolab.platform import NS_TO_SEC
from qibolab.sequence import PulseSequence
from qibolab.sweeper import Parameter, Sweeper, SweeperType

GlobalBackend.set_backend("qibolab", "spinq10q")
platform: Platform = GlobalBackend().platform
controller: QbloxController = platform.instruments["qblox_controller"]

sequence = PulseSequence()
ro_pulses = {}

qid = 0
qubit = platform.qubits[qid]
qubits = {qid: qubit}

ro_pulse = platform.create_qubit_readout_pulse(qid, start=0)
ro_pulse.frequency = int(2e9)
sequence.add(ro_pulse)

freq_width = 2e7
freq_step = 2e5
bias_width = 0.8
bias_step = 0.01
nshots = 1024
relaxation_time = 5000

navgs = nshots
repetition_duration = sequence.finish + relaxation_time

# define the parameters to sweep and their range:
delta_frequency_range = np.arange(-freq_width // 2, freq_width // 2, freq_step)
freq_sweeper = Sweeper(
    Parameter.frequency, delta_frequency_range, [ro_pulse], type=SweeperType.OFFSET
)

delta_bias_range = np.arange(-bias_width / 2, bias_width / 2, bias_step)
bias_sweeper = Sweeper(
    Parameter.bias, delta_bias_range, qubits=[qubit], type=SweeperType.OFFSET
)

options = ExecutionParameters(
    nshots=nshots,
    relaxation_time=relaxation_time,
    acquisition_type=AcquisitionType.INTEGRATION,
    averaging_mode=AveragingMode.CYCLIC,
)
time = (sequence.duration + relaxation_time) * nshots * NS_TO_SEC
sweepers = (bias_sweeper, freq_sweeper)
for sweep in sweepers:
    time *= len(sweep.values)

# mock
controller.is_connected = True


class Sequencers:
    def __getitem__(self, index):
        return self

    def set(self, *args):
        pass


class Device:
    sequencers = Sequencers()


for mod in controller.modules.values():
    mod.device = Device()
    mod._device_num_sequencers = 0
# end mock

# for name, mod in controller.modules.items():
#     if "qcm_rf" in name:
#         continue

mod = controller.modules["qcm_bb0"]
channels = controller._set_module_channel_map(mod, qubits)
pulses = sequence.get_channel_pulses(*channels)
mod.process_pulse_sequence(qubits, pulses, navgs, nshots, repetition_duration, sweepers)
print(mod._sequencers["o1"][0].program)
