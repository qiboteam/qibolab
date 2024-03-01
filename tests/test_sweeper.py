import numpy as np
import pytest

from qibolab.pulses import Pulse, Rectangular
from qibolab.qubits import Qubit
from qibolab.sweeper import Parameter, QubitParameter, Sweeper


@pytest.mark.parametrize("parameter", Parameter)
def test_sweeper_pulses(parameter):
    pulse = Pulse(40, 0.1, int(1e9), 0.0, Rectangular(), "channel")
    if parameter is Parameter.amplitude:
        parameter_range = np.random.rand(10)
    else:
        parameter_range = np.random.randint(10, size=10)
    if parameter in QubitParameter:
        with pytest.raises(ValueError):
            sweeper = Sweeper(parameter, parameter_range, [pulse])
    else:
        sweeper = Sweeper(parameter, parameter_range, [pulse])
        assert sweeper.parameter is parameter


@pytest.mark.parametrize("parameter", Parameter)
def test_sweeper_qubits(parameter):
    qubit = Qubit(0)
    parameter_range = np.random.randint(10, size=10)
    if parameter in QubitParameter:
        sweeper = Sweeper(parameter, parameter_range, qubits=[qubit])
        assert sweeper.parameter is parameter
    else:
        with pytest.raises(ValueError):
            sweeper = Sweeper(parameter, parameter_range, qubits=[qubit])


def test_sweeper_errors():
    pulse = Pulse(40, 0.1, int(1e9), 0.0, Rectangular(), "channel")
    qubit = Qubit(0)
    parameter_range = np.random.randint(10, size=10)
    with pytest.raises(ValueError):
        Sweeper(Parameter.frequency, parameter_range)
    with pytest.raises(ValueError):
        Sweeper(Parameter.frequency, parameter_range, [pulse], [qubit])
