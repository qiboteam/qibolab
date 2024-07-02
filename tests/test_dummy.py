import numpy as np
import pytest

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters, create_platform
from qibolab.pulses import (
    Delay,
    Gaussian,
    GaussianSquare,
    Pulse,
    PulseSequence,
    PulseType,
)
from qibolab.qubits import QubitPair
from qibolab.sweeper import ChannelParameter, Parameter, Sweeper

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
    mz_seq = platform.create_MZ_pulse(0)
    mz_pulse = next(iter(mz_seq.values()))[0]
    sequence = PulseSequence()
    sequence.extend(mz_seq)
    sequence.extend(platform.create_RX12_pulse(0))
    options = ExecutionParameters(nshots=100, acquisition_type=acquisition)
    result = platform.execute_pulse_sequence(sequence, options)
    if acquisition is AcquisitionType.INTEGRATION:
        assert result[mz_pulse.id].magnitude.shape == (nshots,)
    elif acquisition is AcquisitionType.RAW:
        assert result[mz_pulse.id].magnitude.shape == (nshots * mz_seq.duration,)


def test_dummy_execute_coupler_pulse():
    platform = create_platform("dummy_couplers")
    sequence = PulseSequence()

    channel = platform.get_coupler(0).flux
    pulse = Pulse(
        duration=30,
        amplitude=0.05,
        envelope=GaussianSquare(rel_sigma=5, width=0.75),
        type=PulseType.COUPLERFLUX,
    )
    sequence[channel.name].append(pulse)

    options = ExecutionParameters(nshots=None)
    result = platform.execute_pulse_sequence(sequence, options)


def test_dummy_execute_pulse_sequence_couplers():
    platform = create_platform("dummy_couplers")
    qubit_ordered_pair = QubitPair(
        platform.qubits[1], platform.qubits[2], coupler=platform.couplers[1]
    )
    sequence = PulseSequence()

    cz = platform.create_CZ_pulse_sequence(
        qubits=(qubit_ordered_pair.qubit1.name, qubit_ordered_pair.qubit2.name),
    )
    sequence.extend(cz)
    sequence[platform.qubits[0].measure.name].append(Delay(duration=40))
    sequence[platform.qubits[2].measure.name].append(Delay(duration=40))
    sequence.extend(platform.create_MZ_pulse(0))
    sequence.extend(platform.create_MZ_pulse(2))
    options = ExecutionParameters(nshots=None)
    result = platform.execute_pulse_sequence(sequence, options)


@pytest.mark.parametrize("name", PLATFORM_NAMES)
def test_dummy_execute_pulse_sequence_fast_reset(name):
    platform = create_platform(name)
    sequence = PulseSequence()
    sequence.extend(platform.create_MZ_pulse(0))
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
    sequence.extend(platform.create_MZ_pulse(0))
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
    mz_seq = platform.create_MZ_pulse(qubit=0)
    pulse = next(iter(mz_seq.values()))[0]

    parameter_range = np.random.randint(SWEPT_POINTS, size=SWEPT_POINTS)
    sequence.extend(mz_seq)
    sweeper = Sweeper(
        Parameter.frequency,
        parameter_range,
        channels=[platform.get_qubit(0).measure.name],
    )
    options = ExecutionParameters(
        nshots=10,
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.RAW,
    )
    results = platform.sweep(sequence, options, sweeper)
    assert pulse.id in results
    shape = results[pulse.id].magnitude.shape
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
    mz_seq = platform.create_MZ_pulse(qubit=0)
    mz_pulse = next(iter(mz_seq.values()))[0]
    coupler_pulse = Pulse.flux(
        duration=40,
        amplitude=0.5,
        envelope=GaussianSquare(rel_sigma=0.2, width=0.75),
        type=PulseType.COUPLERFLUX,
    )
    sequence.extend(mz_seq)
    sequence[platform.get_coupler(0).flux.name].append(coupler_pulse)
    if parameter is Parameter.amplitude:
        parameter_range = np.random.rand(SWEPT_POINTS)
    else:
        parameter_range = np.random.randint(SWEPT_POINTS, size=SWEPT_POINTS)
    if parameter in ChannelParameter:
        sweeper = Sweeper(
            parameter, parameter_range, channels=[platform.couplers[0].flux.name]
        )
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

    assert mz_pulse.id in results
    if average:
        results_shape = (
            results[mz_pulse.id].magnitude.shape
            if acquisition is AcquisitionType.INTEGRATION
            else results[mz_pulse.id].statistical_frequency.shape
        )
    else:
        results_shape = (
            results[mz_pulse.id].magnitude.shape
            if acquisition is AcquisitionType.INTEGRATION
            else results[mz_pulse.id].samples.shape
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
    mz_seq = platform.create_MZ_pulse(qubit=0)
    pulse = next(iter(mz_seq.values()))[0]
    if parameter is Parameter.amplitude:
        parameter_range = np.random.rand(SWEPT_POINTS)
    else:
        parameter_range = np.random.randint(SWEPT_POINTS, size=SWEPT_POINTS)
    sequence.extend(mz_seq)
    if parameter in ChannelParameter:
        channel = (
            platform.qubits[0].drive.name
            if parameter is Parameter.frequency
            else platform.qubits[0].flux.name
        )
        sweeper = Sweeper(parameter, parameter_range, channels=[channel])
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

    assert pulse.id in results
    if average:
        results_shape = (
            results[pulse.id].magnitude.shape
            if acquisition is AcquisitionType.INTEGRATION
            else results[pulse.id].statistical_frequency.shape
        )
    else:
        results_shape = (
            results[pulse.id].magnitude.shape
            if acquisition is AcquisitionType.INTEGRATION
            else results[pulse.id].samples.shape
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
    pulse = Pulse(
        duration=40, amplitude=0.1, envelope=Gaussian(rel_sigma=5), type=PulseType.DRIVE
    )
    mz_seq = platform.create_MZ_pulse(qubit=0)
    mz_pulse = next(iter(mz_seq.values()))[0]
    sequence[platform.get_qubit(0).drive.name].append(pulse)
    sequence[platform.qubits[0].measure.name].append(Delay(duration=pulse.duration))
    sequence.extend(mz_seq)
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

    if parameter1 in ChannelParameter:
        channel = (
            platform.qubits[0].measure.name
            if parameter1 is Parameter.frequency
            else platform.qubits[0].flux.name
        )
        sweeper1 = Sweeper(parameter1, parameter_range_1, channels=[channel])
    else:
        sweeper1 = Sweeper(parameter1, parameter_range_1, pulses=[mz_pulse])
    if parameter2 in ChannelParameter:
        sweeper2 = Sweeper(
            parameter2, parameter_range_2, channels=[platform.qubits[0].flux.name]
        )
    else:
        sweeper2 = Sweeper(parameter2, parameter_range_2, pulses=[pulse])

    options = ExecutionParameters(
        nshots=nshots,
        averaging_mode=average,
        acquisition_type=acquisition,
    )
    average = not options.averaging_mode is AveragingMode.SINGLESHOT
    results = platform.sweep(sequence, options, sweeper1, sweeper2)

    assert mz_pulse.id in results

    if average:
        results_shape = (
            results[mz_pulse.id].magnitude.shape
            if acquisition is AcquisitionType.INTEGRATION
            else results[mz_pulse.id].statistical_frequency.shape
        )
    else:
        results_shape = (
            results[mz_pulse.id].magnitude.shape
            if acquisition is AcquisitionType.INTEGRATION
            else results[mz_pulse.id].samples.shape
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
    mz_pulses = {}
    for qubit in platform.qubits:
        mz_seq = platform.create_MZ_pulse(qubit=qubit)
        mz_pulses[qubit] = next(iter(mz_seq.values()))[0]
        sequence.extend(mz_seq)
    parameter_range = (
        np.random.rand(SWEPT_POINTS)
        if parameter is Parameter.amplitude
        else np.random.randint(SWEPT_POINTS, size=SWEPT_POINTS)
    )

    if parameter in ChannelParameter:
        sweeper1 = Sweeper(
            parameter,
            parameter_range,
            channels=[qubit.measure.name for qubit in platform.qubits.values()],
        )
    else:
        sweeper1 = Sweeper(
            parameter,
            parameter_range,
            pulses=[mz_pulses[qubit] for qubit in platform.qubits],
        )

    options = ExecutionParameters(
        nshots=nshots,
        averaging_mode=average,
        acquisition_type=acquisition,
    )
    average = not options.averaging_mode is AveragingMode.SINGLESHOT
    results = platform.sweep(sequence, options, sweeper1)

    for pulse in mz_pulses.values():
        assert pulse.id in results
        if average:
            results_shape = (
                results[pulse.id].magnitude.shape
                if acquisition is AcquisitionType.INTEGRATION
                else results[pulse.id].statistical_frequency.shape
            )
        else:
            results_shape = (
                results[pulse.id].magnitude.shape
                if acquisition is AcquisitionType.INTEGRATION
                else results[pulse.id].samples.shape
            )
        assert results_shape == (SWEPT_POINTS,) if average else (nshots, SWEPT_POINTS)


# TODO: add test_dummy_double_sweep_multiplex
