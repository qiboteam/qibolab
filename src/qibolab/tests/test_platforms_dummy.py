import numpy as np
import pytest

from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper


def test_dummy_initialization():
    platform = Platform("dummy")
    platform.reload_settings()
    platform.connect()
    platform.setup()
    platform.start()
    platform.stop()
    platform.disconnect()


def test_dummy_execute_pulse_sequence():
    platform = Platform("dummy")
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_readout_pulse(0, 0))
    result = platform.execute_pulse_sequence(sequence, nshots=100)


# @pytest.mark.parametrize("parameter", ["frequency", "amplitude", "attenuation", "gain", current])
@pytest.mark.parametrize("average", [True, False])
def test_dummy_single_sweep(average):
    platform = Platform("dummy")
    sequence = PulseSequence()
    pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
    for parameter in Parameter:
        if parameter is Parameter.amplitude:
            parameter_range = np.random.rand(10)
        else:
            parameter_range = np.random.randint(10, size=10)
        sequence.add(pulse)
        sweeper = Sweeper(parameter, parameter_range, [pulse])
        platform.sweep(sequence, sweeper, average=average)


@pytest.mark.parametrize("average", [True, False])
def test_dummy_double_sweep(average):
    platform = Platform("dummy")
    sweepers = []
    sequence = PulseSequence()
    pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
    sequence.add(pulse)
    for _ in range(2):
        for parameter in Parameter:
            if parameter is Parameter.amplitude:
                parameter_range = np.random.rand(10)
            else:
                parameter_range = np.random.randint(10, size=10)
        sweepers.append(Sweeper(parameter, parameter_range, [pulse]))
    platform.sweep(sequence, *sweepers, average=average)
