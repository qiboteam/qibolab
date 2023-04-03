import numpy as np
import pytest

from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, QubitParameter, Sweeper


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
@pytest.mark.parametrize("nshots", [100, 200])
def test_dummy_single_sweep(parameter, average, nshots):
    swept_points = 5
    platform = Platform("dummy")
    sequence = PulseSequence()
    pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
    if parameter is Parameter.amplitude:
        parameter_range = np.random.rand(swept_points)
    else:
        parameter_range = np.random.randint(swept_points, size=swept_points)
    sequence.add(pulse)
    if parameter in QubitParameter:
        sweeper = Sweeper(parameter, parameter_range, qubits=[platform.qubits[0]])
    else:
        sweeper = Sweeper(parameter, parameter_range, pulses=[pulse])
    results = platform.sweep(sequence, sweeper, average=average, nshots=nshots)

    assert pulse.serial and pulse.qubit in results
    assert len(results[pulse.qubit]) == swept_points if average else int(nshots * swept_points)


@pytest.mark.parametrize("parameter1", Parameter)
@pytest.mark.parametrize("parameter2", Parameter)
@pytest.mark.parametrize("average", [True, False])
@pytest.mark.parametrize("nshots", [100, 1000])
def test_dummy_double_sweep(parameter1, parameter2, average, nshots):
    swept_points = 5
    platform = Platform("dummy")
    sequence = PulseSequence()
    pulse = platform.create_qubit_drive_pulse(qubit=0, start=0, duration=1000)
    ro_pulse = platform.create_qubit_readout_pulse(qubit=0, start=pulse.finish)
    sequence.add(pulse)
    sequence.add(ro_pulse)
    parameter_range_1 = (
        np.random.rand(swept_points)
        if parameter1 is Parameter.amplitude
        else np.random.randint(swept_points, size=swept_points)
    )
    parameter_range_2 = (
        np.random.rand(swept_points)
        if parameter2 is Parameter.amplitude
        else np.random.randint(swept_points, size=swept_points)
    )

    if parameter1 in QubitParameter:
        sweeper1 = Sweeper(parameter1, parameter_range_1, qubits=[platform.qubits[0]])
    else:
        sweeper1 = Sweeper(parameter1, parameter_range_1, pulses=[ro_pulse])
    if parameter2 in QubitParameter:
        sweeper2 = Sweeper(parameter2, parameter_range_2, qubits=[platform.qubits[0]])
    else:
        sweeper2 = Sweeper(parameter2, parameter_range_2, pulses=[pulse])
    results = platform.sweep(sequence, sweeper1, sweeper2, average=average, nshots=nshots)

    assert ro_pulse.serial and ro_pulse.qubit in results
    assert len(results[ro_pulse.serial]) == swept_points**2 if average else int(nshots * swept_points**2)


@pytest.mark.parametrize("parameter", Parameter)
@pytest.mark.parametrize("average", [True, False])
@pytest.mark.parametrize("nshots", [100, 1000])
def test_dummy_single_sweep_multiplex(parameter, average, nshots):
    swept_points = 5
    platform = Platform("dummy")
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in platform.qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit=qubit, start=0)
        sequence.add(ro_pulses[qubit])
    parameter_range = (
        np.random.rand(swept_points)
        if parameter is Parameter.amplitude
        else np.random.randint(swept_points, size=swept_points)
    )

    if parameter in QubitParameter:
        sweeper1 = Sweeper(parameter, parameter_range, qubits=[platform.qubits[qubit] for qubit in platform.qubits])
    else:
        sweeper1 = Sweeper(parameter, parameter_range, pulses=[ro_pulses[qubit] for qubit in platform.qubits])

    results = platform.sweep(sequence, sweeper1, average=average, nshots=nshots)

    for ro_pulse in ro_pulses.values():
        assert ro_pulse.serial and ro_pulse.qubit in results
        assert len(results[ro_pulse.qubit]) == swept_points if average else int(nshots * swept_points)


# TODO: add test_dummy_double_sweep_multiplex
