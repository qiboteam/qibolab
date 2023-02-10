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


@pytest.mark.parametrize("parameter", Parameter)
@pytest.mark.parametrize("average", [True, False])
@pytest.mark.parametrize("nshots", [100, 1024, 2048])
def test_dummy_single_sweep(parameter, average, nshots):
    swept_points = 10
    platform = Platform("dummy")
    sequence = PulseSequence()
    pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
    if parameter is Parameter.amplitude:
        parameter_range = np.random.rand(10)
    else:
        parameter_range = np.random.randint(10, size=10)
    sequence.add(pulse)
    sweeper = Sweeper(parameter, parameter_range, [pulse])
    results = platform.sweep(sequence, sweeper, average=average, nshots=nshots)

    assert pulse.serial and pulse.qubit in results
    assert len(results[pulse.serial]) == swept_points if average else int(nshots * swept_points)


@pytest.mark.parametrize("parameter1", Parameter)
@pytest.mark.parametrize("parameter2", Parameter)
@pytest.mark.parametrize("average", [True, False])
@pytest.mark.parametrize("nshots", [100, 100])
def test_dummy_double_sweep(parameter1, parameter2, average, nshots):
    swept_points = 10
    platform = Platform("dummy")
    sequence = PulseSequence()
    pulse = platform.create_qubit_drive_pulse(qubit=0, start=0, duration=1000)
    ro_pulse = platform.create_qubit_readout_pulse(qubit=0, start=pulse.finish)
    sequence.add(pulse)
    sequence.add(ro_pulse)
    parameter_range_1 = np.random.rand(10) if parameter1 is Parameter.amplitude else np.random.randint(10, size=10)
    parameter_range_2 = np.random.rand(10) if parameter2 is Parameter.amplitude else np.random.randint(10, size=10)

    sweeper1 = Sweeper(parameter1, parameter_range_1, [ro_pulse])
    sweeper2 = Sweeper(parameter2, parameter_range_2, [pulse])
    results = platform.sweep(sequence, sweeper1, sweeper2, average=average, nshots=nshots)

    assert ro_pulse.serial and ro_pulse.qubit in results
    assert len(results[ro_pulse.serial]) == swept_points**2 if average else int(nshots * swept_points**2)
