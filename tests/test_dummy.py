import numpy as np
import pytest

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters, create_platform
from qibolab.pulses import CouplerFluxPulse, PulseSequence
from qibolab.qubits import QubitPair
from qibolab.sweeper import Parameter, QubitParameter, Sweeper

SWEPT_POINTS = 5
PLATFORM_NAMES = ["dummy", "dummy_couplers"]


@pytest.mark.parametrize("name", PLATFORM_NAMES)
def test_dummy_initialization(name):
    platform = create_platform(name)
    platform.connect()
    platform.disconnect()


@pytest.mark.parametrize("name", PLATFORM_NAMES)
@pytest.mark.parametrize(
    "acquisition", [AcquisitionType.INTEGRATION, AcquisitionType.RAW]
)
def test_dummy_execute_pulse_sequence(name, acquisition):
    nshots = 100
    platform = create_platform(name)
    ro_pulse = platform.create_qubit_readout_pulse(0, 0)
    sequence = PulseSequence()
    sequence.append(platform.create_qubit_readout_pulse(0, 0))
    sequence.append(platform.create_RX12_pulse(0, 0))
    options = ExecutionParameters(nshots=100, acquisition_type=acquisition)
    result = platform.execute_pulse_sequence(sequence, options)
    if acquisition is AcquisitionType.INTEGRATION:
        assert result[0].magnitude.shape == (nshots,)
    elif acquisition is AcquisitionType.RAW:
        assert result[0].magnitude.shape == (nshots * ro_pulse.duration,)


def test_dummy_execute_coupler_pulse():
    platform = create_platform("dummy_couplers")
    sequence = PulseSequence()

    pulse = platform.create_coupler_pulse(coupler=0, start=0)
    sequence.append(pulse)

    options = ExecutionParameters(nshots=None)
    result = platform.execute_pulse_sequence(sequence, options)


def test_dummy_execute_pulse_sequence_couplers():
    platform = create_platform("dummy_couplers")
    qubit_ordered_pair = QubitPair(
        platform.qubits[1], platform.qubits[2], platform.couplers[1]
    )
    sequence = PulseSequence()

    cz, cz_phases = platform.create_CZ_pulse_sequence(
        qubits=(qubit_ordered_pair.qubit1.name, qubit_ordered_pair.qubit2.name),
        start=0,
    )
    sequence.extend(cz.get_qubit_pulses(qubit_ordered_pair.qubit1.name))
    sequence.extend(cz.get_qubit_pulses(qubit_ordered_pair.qubit2.name))
    sequence.extend(cz.coupler_pulses(qubit_ordered_pair.coupler.name))
    sequence.append(platform.create_qubit_readout_pulse(0, 40))
    sequence.append(platform.create_qubit_readout_pulse(2, 40))
    options = ExecutionParameters(nshots=None)
    result = platform.execute_pulse_sequence(sequence, options)

    test_phases = {1: 0.0, 2: 0.0}

    assert test_phases == cz_phases


@pytest.mark.parametrize("name", PLATFORM_NAMES)
def test_dummy_execute_pulse_sequence_fast_reset(name):
    platform = create_platform(name)
    sequence = PulseSequence()
    sequence.append(platform.create_qubit_readout_pulse(0, 0))
    options = ExecutionParameters(nshots=None, fast_reset=True)
    result = platform.execute_pulse_sequence(sequence, options)


@pytest.mark.parametrize("name", PLATFORM_NAMES)
@pytest.mark.parametrize(
    "acquisition", [AcquisitionType.INTEGRATION, AcquisitionType.DISCRIMINATION]
)
@pytest.mark.parametrize("batch_size", [None, 3, 5])
def test_dummy_execute_pulse_sequence_unrolling(name, acquisition, batch_size):
    nshots = 100
    nsequences = 10
    platform = create_platform(name)
    platform.instruments["dummy"].UNROLLING_BATCH_SIZE = batch_size
    sequences = []
    sequence = PulseSequence()
    sequence.append(platform.create_qubit_readout_pulse(0, 0))
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


@pytest.mark.parametrize("name", PLATFORM_NAMES)
def test_dummy_single_sweep_raw(name):
    platform = create_platform(name)
    sequence = PulseSequence()
    pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)

    parameter_range = np.random.randint(SWEPT_POINTS, size=SWEPT_POINTS)
    sequence.append(pulse)
    sweeper = Sweeper(Parameter.frequency, parameter_range, pulses=[pulse])
    options = ExecutionParameters(
        nshots=10,
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.RAW,
    )
    results = platform.sweep(sequence, options, sweeper)
    assert pulse.id and pulse.qubit in results
    shape = results[pulse.qubit].magnitude.shape
    assert shape == (pulse.duration * SWEPT_POINTS,)


@pytest.mark.parametrize("fast_reset", [True, False])
@pytest.mark.parametrize(
    "parameter", [Parameter.amplitude, Parameter.duration, Parameter.bias]
)
@pytest.mark.parametrize("average", [AveragingMode.SINGLESHOT, AveragingMode.CYCLIC])
@pytest.mark.parametrize(
    "acquisition", [AcquisitionType.INTEGRATION, AcquisitionType.DISCRIMINATION]
)
@pytest.mark.parametrize("nshots", [10, 20])
def test_dummy_single_sweep_coupler(
    fast_reset, parameter, average, acquisition, nshots
):
    platform = create_platform("dummy_couplers")
    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
    coupler_pulse = CouplerFluxPulse(
        start=0,
        duration=40,
        amplitude=0.5,
        shape="GaussianSquare(5, 0.75)",
        channel="flux_coupler-0",
        qubit=0,
    )
    if parameter is Parameter.amplitude:
        parameter_range = np.random.rand(SWEPT_POINTS)
    else:
        parameter_range = np.random.randint(SWEPT_POINTS, size=SWEPT_POINTS)
    sequence.append(ro_pulse)
    if parameter in QubitParameter:
        sweeper = Sweeper(parameter, parameter_range, couplers=[platform.couplers[0]])
    else:
        sweeper = Sweeper(parameter, parameter_range, pulses=[coupler_pulse])
    options = ExecutionParameters(
        nshots=nshots,
        averaging_mode=average,
        acquisition_type=acquisition,
        fast_reset=fast_reset,
    )
    average = not options.averaging_mode is AveragingMode.SINGLESHOT
    results = platform.sweep(sequence, options, sweeper)

    assert ro_pulse.id and ro_pulse.qubit in results
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


@pytest.mark.parametrize("name", PLATFORM_NAMES)
@pytest.mark.parametrize("fast_reset", [True, False])
@pytest.mark.parametrize("parameter", Parameter)
@pytest.mark.parametrize("average", [AveragingMode.SINGLESHOT, AveragingMode.CYCLIC])
@pytest.mark.parametrize(
    "acquisition", [AcquisitionType.INTEGRATION, AcquisitionType.DISCRIMINATION]
)
@pytest.mark.parametrize("nshots", [10, 20])
def test_dummy_single_sweep(name, fast_reset, parameter, average, acquisition, nshots):
    platform = create_platform(name)
    sequence = PulseSequence()
    pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
    if parameter is Parameter.amplitude:
        parameter_range = np.random.rand(SWEPT_POINTS)
    else:
        parameter_range = np.random.randint(SWEPT_POINTS, size=SWEPT_POINTS)
    sequence.append(pulse)
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

    assert pulse.id and pulse.qubit in results
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


@pytest.mark.parametrize("name", PLATFORM_NAMES)
@pytest.mark.parametrize("parameter1", Parameter)
@pytest.mark.parametrize("parameter2", Parameter)
@pytest.mark.parametrize("average", [AveragingMode.SINGLESHOT, AveragingMode.CYCLIC])
@pytest.mark.parametrize(
    "acquisition", [AcquisitionType.INTEGRATION, AcquisitionType.DISCRIMINATION]
)
@pytest.mark.parametrize("nshots", [10, 20])
def test_dummy_double_sweep(name, parameter1, parameter2, average, acquisition, nshots):
    platform = create_platform(name)
    sequence = PulseSequence()
    pulse = platform.create_qubit_drive_pulse(qubit=0, start=0, duration=1000)
    ro_pulse = platform.create_qubit_readout_pulse(qubit=0, start=pulse.finish)
    sequence.append(pulse)
    sequence.append(ro_pulse)
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

    assert ro_pulse.id and ro_pulse.qubit in results

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

    assert (
        results_shape == (SWEPT_POINTS, SWEPT_POINTS)
        if average
        else (nshots, SWEPT_POINTS, SWEPT_POINTS)
    )


@pytest.mark.parametrize("name", PLATFORM_NAMES)
@pytest.mark.parametrize("parameter", Parameter)
@pytest.mark.parametrize("average", [AveragingMode.SINGLESHOT, AveragingMode.CYCLIC])
@pytest.mark.parametrize(
    "acquisition", [AcquisitionType.INTEGRATION, AcquisitionType.DISCRIMINATION]
)
@pytest.mark.parametrize("nshots", [10, 20])
def test_dummy_single_sweep_multiplex(name, parameter, average, acquisition, nshots):
    platform = create_platform(name)
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in platform.qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit=qubit, start=0)
        sequence.append(ro_pulses[qubit])
    parameter_range = (
        np.random.rand(SWEPT_POINTS)
        if parameter is Parameter.amplitude
        else np.random.randint(SWEPT_POINTS, size=SWEPT_POINTS)
    )

    if parameter in QubitParameter:
        sweeper1 = Sweeper(
            parameter,
            parameter_range,
            qubits=[platform.qubits[qubit] for qubit in platform.qubits],
        )
    else:
        sweeper1 = Sweeper(
            parameter,
            parameter_range,
            pulses=[ro_pulses[qubit] for qubit in platform.qubits],
        )

    options = ExecutionParameters(
        nshots=nshots,
        averaging_mode=average,
        acquisition_type=acquisition,
    )
    average = not options.averaging_mode is AveragingMode.SINGLESHOT
    results = platform.sweep(sequence, options, sweeper1)

    for ro_pulse in ro_pulses.values():
        assert ro_pulse.id and ro_pulse.qubit in results
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
