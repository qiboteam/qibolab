import numpy as np
import pytest

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters, create_platform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, QubitParameter, Sweeper

SWEPT_POINTS = 5


def test_dummy_initialization():
    platform = create_platform("dummy")
    platform.connect()
    platform.setup()
    platform.start()
    platform.stop()
    platform.disconnect()


@pytest.mark.parametrize("acquisition", [AcquisitionType.INTEGRATION, AcquisitionType.RAW])
def test_dummy_execute_pulse_sequence(acquisition):
    nshots = 100
    platform = create_platform("dummy")
    ro_pulse = platform.create_qubit_readout_pulse(0, 0)
    sequence = PulseSequence()
    sequence.add(ro_pulse)
    options = ExecutionParameters(nshots=nshots, acquisition_type=acquisition)
    result = platform.execute_pulse_sequence(sequence, options)
    if acquisition is AcquisitionType.INTEGRATION:
        assert result[0].magnitude.shape == (nshots,)
    elif acquisition is AcquisitionType.RAW:
        assert result[0].magnitude.shape == (nshots * ro_pulse.duration,)


def test_dummy_execute_pulse_sequence_fast_reset():
    platform = create_platform("dummy")
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_readout_pulse(0, 0))
    options = ExecutionParameters(nshots=None, fast_reset=True)
    result = platform.execute_pulse_sequence(sequence, options)


@pytest.mark.parametrize("acquisition", [AcquisitionType.INTEGRATION, AcquisitionType.DISCRIMINATION])
@pytest.mark.parametrize("batch_size", [None, 3, 5])
def test_dummy_execute_pulse_sequence_unrolling(acquisition, batch_size):
    nshots = 100
    nsequences = 10
    platform = create_platform("dummy")
    platform.instruments["dummy"].UNROLLING_BATCH_SIZE = batch_size
    sequences = []
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_readout_pulse(0, 0))
    for _ in range(nsequences):
        sequences.append(sequence)
    options = ExecutionParameters(nshots=nshots, acquisition_type=acquisition)
    result = platform.execute_pulse_sequences(sequences, options)
    assert len(result[0]) == nsequences
    for r in result[0]:
        if acquisition is AcquisitionType.INTEGRATION:
            assert r.magnitude.shape == (nshots,)
        if acquisition is AcquisitionType.DISCRIMINATION:
            assert r.samples.shape == (nshots,)


def test_dummy_single_sweep_RAW():
    platform = create_platform("dummy")
    sequence = PulseSequence()
    pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)

    parameter_range = np.random.randint(SWEPT_POINTS, size=SWEPT_POINTS)
    sequence.add(pulse)
    sweeper = Sweeper(Parameter.frequency, parameter_range, pulses=[pulse])
    options = ExecutionParameters(
        nshots=10,
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.RAW,
    )
    results = platform.sweep(sequence, options, sweeper)
    assert pulse.serial and pulse.qubit in results
    shape = results[pulse.qubit].magnitude.shape
    samples = platform.settings.sampling_rate * 1e-9 * pulse.duration

    assert shape == (samples * SWEPT_POINTS,)


@pytest.mark.parametrize("fast_reset", [True, False])
@pytest.mark.parametrize("parameter", Parameter)
@pytest.mark.parametrize("average", [AveragingMode.SINGLESHOT, AveragingMode.CYCLIC])
@pytest.mark.parametrize("acquisition", [AcquisitionType.INTEGRATION, AcquisitionType.DISCRIMINATION])
@pytest.mark.parametrize("nshots", [10, 20])
def test_dummy_single_sweep(fast_reset, parameter, average, acquisition, nshots):
    platform = create_platform("dummy")
    sequence = PulseSequence()
    pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
    if parameter is Parameter.amplitude:
        parameter_range = np.random.rand(SWEPT_POINTS)
    else:
        parameter_range = np.random.randint(SWEPT_POINTS, size=SWEPT_POINTS)
    sequence.add(pulse)
    if parameter in QubitParameter:
        sweeper = Sweeper(parameter, parameter_range, qubits=[platform.qubits[0]])
    else:
        sweeper = Sweeper(parameter, parameter_range, pulses=[pulse])
    options = ExecutionParameters(
        nshots=nshots,
        averaging_mode=average,
        acquisition_type=acquisition,
        fast_reset=fast_reset,
    )
    average = not options.averaging_mode is AveragingMode.SINGLESHOT
    results = platform.sweep(sequence, options, sweeper)

    assert pulse.serial and pulse.qubit in results
    if average:
        results_shape = (
            results[pulse.qubit].magnitude.shape
            if acquisition is AcquisitionType.INTEGRATION
            else results[pulse.qubit].statistical_frequency.shape
        )
    else:
        results_shape = (
            results[pulse.qubit].magnitude.shape
            if acquisition is AcquisitionType.INTEGRATION
            else results[pulse.qubit].samples.shape
        )
    assert results_shape == (SWEPT_POINTS,) if average else (nshots, SWEPT_POINTS)


@pytest.mark.parametrize("parameter1", Parameter)
@pytest.mark.parametrize("parameter2", Parameter)
@pytest.mark.parametrize("average", [AveragingMode.SINGLESHOT, AveragingMode.CYCLIC])
@pytest.mark.parametrize("acquisition", [AcquisitionType.INTEGRATION, AcquisitionType.DISCRIMINATION])
@pytest.mark.parametrize("nshots", [10, 20])
def test_dummy_double_sweep(parameter1, parameter2, average, acquisition, nshots):
    platform = create_platform("dummy")
    sequence = PulseSequence()
    pulse = platform.create_qubit_drive_pulse(qubit=0, start=0, duration=1000)
    ro_pulse = platform.create_qubit_readout_pulse(qubit=0, start=pulse.finish)
    sequence.add(pulse)
    sequence.add(ro_pulse)
    parameter_range_1 = (
        np.random.rand(SWEPT_POINTS)
        if parameter1 is Parameter.amplitude
        else np.random.randint(SWEPT_POINTS, size=SWEPT_POINTS)
    )
    parameter_range_2 = (
        np.random.rand(SWEPT_POINTS)
        if parameter2 is Parameter.amplitude
        else np.random.randint(SWEPT_POINTS, size=SWEPT_POINTS)
    )

    if parameter1 in QubitParameter:
        sweeper1 = Sweeper(parameter1, parameter_range_1, qubits=[platform.qubits[0]])
    else:
        sweeper1 = Sweeper(parameter1, parameter_range_1, pulses=[ro_pulse])
    if parameter2 in QubitParameter:
        sweeper2 = Sweeper(parameter2, parameter_range_2, qubits=[platform.qubits[0]])
    else:
        sweeper2 = Sweeper(parameter2, parameter_range_2, pulses=[pulse])

    options = ExecutionParameters(
        nshots=nshots,
        averaging_mode=average,
        acquisition_type=acquisition,
    )
    average = not options.averaging_mode is AveragingMode.SINGLESHOT
    results = platform.sweep(sequence, options, sweeper1, sweeper2)

    assert ro_pulse.serial and ro_pulse.qubit in results

    if average:
        results_shape = (
            results[pulse.qubit].magnitude.shape
            if acquisition is AcquisitionType.INTEGRATION
            else results[pulse.qubit].statistical_frequency.shape
        )
    else:
        results_shape = (
            results[pulse.qubit].magnitude.shape
            if acquisition is AcquisitionType.INTEGRATION
            else results[pulse.qubit].samples.shape
        )

    assert results_shape == (SWEPT_POINTS, SWEPT_POINTS) if average else (nshots, SWEPT_POINTS, SWEPT_POINTS)


@pytest.mark.parametrize("parameter", Parameter)
@pytest.mark.parametrize("average", [AveragingMode.SINGLESHOT, AveragingMode.CYCLIC])
@pytest.mark.parametrize("acquisition", [AcquisitionType.INTEGRATION, AcquisitionType.DISCRIMINATION])
@pytest.mark.parametrize("nshots", [10, 20])
def test_dummy_single_sweep_multiplex(parameter, average, acquisition, nshots):
    platform = create_platform("dummy")
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in platform.qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit=qubit, start=0)
        sequence.add(ro_pulses[qubit])
    parameter_range = (
        np.random.rand(SWEPT_POINTS)
        if parameter is Parameter.amplitude
        else np.random.randint(SWEPT_POINTS, size=SWEPT_POINTS)
    )

    if parameter in QubitParameter:
        sweeper1 = Sweeper(parameter, parameter_range, qubits=[platform.qubits[qubit] for qubit in platform.qubits])
    else:
        sweeper1 = Sweeper(parameter, parameter_range, pulses=[ro_pulses[qubit] for qubit in platform.qubits])

    options = ExecutionParameters(
        nshots=nshots,
        averaging_mode=average,
        acquisition_type=acquisition,
    )
    average = not options.averaging_mode is AveragingMode.SINGLESHOT
    results = platform.sweep(sequence, options, sweeper1)

    for ro_pulse in ro_pulses.values():
        assert ro_pulse.serial and ro_pulse.qubit in results
        if average:
            results_shape = (
                results[ro_pulse.qubit].magnitude.shape
                if acquisition is AcquisitionType.INTEGRATION
                else results[ro_pulse.qubit].statistical_frequency.shape
            )
        else:
            results_shape = (
                results[ro_pulse.qubit].magnitude.shape
                if acquisition is AcquisitionType.INTEGRATION
                else results[ro_pulse.qubit].samples.shape
            )
        assert results_shape == (SWEPT_POINTS,) if average else (nshots, SWEPT_POINTS)


# TODO: add test_dummy_double_sweep_multiplex
