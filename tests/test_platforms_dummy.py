import numpy as np
import pytest

from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Sweeper


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


@pytest.mark.parametrize("parameter", ["frequency", "amplitude", "attenuation", "gain"])
@pytest.mark.parametrize("average", [True, False])
def test_dummy_single_sweep(parameter, average):
    platform = Platform("dummy")
    sequence = PulseSequence()
    pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
    if parameter == "amplitude":
        parameter_range = np.random.rand(10)
    else:
        parameter_range = np.random.randint(10, size=10)
    sequence.add(pulse)
    sweeper = Sweeper(parameter, parameter_range, [pulse])
    platform.sweep(sequence, sweeper, average=average)


@pytest.mark.parametrize("parameter1", ["frequency", "amplitude", "attenuation", "gain"])
@pytest.mark.parametrize("parameter2", ["frequency", "amplitude", "attenuation", "gain"])
@pytest.mark.parametrize("average", [True, False])
def test_dummy_double_sweep(parameter1, parameter2, average):
    platform = Platform("dummy")
    sequence = PulseSequence()
    pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
    sequence.add(pulse)
    for parameter in [parameter1, parameter2]:
        if parameter == "amplitude":
            parameter_range = np.random.rand(10)
        else:
            parameter_range = np.random.randint(10, size=10)
        sweeper = Sweeper(parameter, parameter_range, [pulse])
    platform.sweep(sequence, sweeper, average=average)
