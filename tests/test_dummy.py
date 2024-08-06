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
    probe_seq = platform.qubits[0].native_gates.MZ.create_sequence()
    probe_pulse = next(iter(probe_seq.values()))[0]
    sequence = PulseSequence()
    sequence.extend(probe_seq)
    sequence.extend(platform.qubits[0].native_gates.RX12.create_sequence())
    options = ExecutionParameters(nshots=100, acquisition_type=acquisition)
    result = platform.execute([sequence], options)
    if acquisition is AcquisitionType.INTEGRATION:
        assert result[probe_pulse.id][0].shape == (nshots, 2)
    elif acquisition is AcquisitionType.RAW:
        assert result[probe_pulse.id][0].shape == (nshots, int(probe_seq.duration), 2)


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
    _ = platform.execute([sequence], options)


def test_dummy_execute_pulse_sequence_couplers():
    platform = create_platform("dummy_couplers")
    sequence = PulseSequence()

    cz = platform.pairs[(1, 2)].native_gates.CZ.create_sequence()

    sequence.extend(cz)
    sequence[platform.qubits[0].probe.name].append(Delay(duration=40))
    sequence[platform.qubits[2].probe.name].append(Delay(duration=40))
    sequence.extend(platform.qubits[0].native_gates.MZ.create_sequence())
    sequence.extend(platform.qubits[2].native_gates.MZ.create_sequence())
    options = ExecutionParameters(nshots=None)
    _ = platform.execute([sequence], options)


@pytest.mark.parametrize("name", PLATFORM_NAMES)
def test_dummy_execute_pulse_sequence_fast_reset(name):
    platform = create_platform(name)
    sequence = PulseSequence()
    sequence.extend(platform.qubits[0].native_gates.MZ.create_sequence())
    options = ExecutionParameters(nshots=None, fast_reset=True)
    _ = platform.execute([sequence], options)


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
    sequence.extend(platform.qubits[0].native_gates.MZ.create_sequence())
    for _ in range(nsequences):
        sequences.append(sequence)
    options = ExecutionParameters(nshots=nshots, acquisition_type=acquisition)
    result = platform.execute(sequences, options)
    assert len(next(iter(result.values()))) == nsequences
    for r in result[0]:
        if acquisition is AcquisitionType.INTEGRATION:
            assert r.magnitude.shape == (nshots,)
        if acquisition is AcquisitionType.DISCRIMINATION:
            assert r.samples.shape == (nshots,)


@pytest.mark.parametrize("name", PLATFORM_NAMES)
def test_dummy_single_sweep_raw(name):
    platform = create_platform(name)
    sequence = PulseSequence()
    probe_seq = platform.qubits[0].native_gates.MZ.create_sequence()
    pulse = next(iter(probe_seq.values()))[0]

    parameter_range = np.random.randint(SWEPT_POINTS, size=SWEPT_POINTS)
    sequence.extend(probe_seq)
    sweeper = Sweeper(
        Parameter.frequency,
        parameter_range,
        channels=[platform.get_qubit(0).probe.name],
    )
    options = ExecutionParameters(
        nshots=10,
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.RAW,
    )
    results = platform.execute([sequence], options, [[sweeper]])
    assert pulse.id in results
    shape = results[pulse.id][0].shape
    assert shape == (SWEPT_POINTS, int(pulse.duration), 2)


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
    probe_seq = platform.qubits[0].native_gates.MZ.create_sequence()
    probe_pulse = next(iter(probe_seq.values()))[0]
    coupler_pulse = Pulse.flux(
        duration=40,
        amplitude=0.5,
        envelope=GaussianSquare(rel_sigma=0.2, width=0.75),
        type=PulseType.COUPLERFLUX,
    )
    sequence.extend(probe_seq)
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
    results = platform.execute([sequence], options, [[sweeper]])

    assert probe_pulse.id in results
    if average:
        results_shape = (
            results[probe_pulse.id][0].shape
            if acquisition is AcquisitionType.INTEGRATION
            else results[probe_pulse.id][0].shape
        )
    else:
        results_shape = (
            results[probe_pulse.id][0].shape
            if acquisition is AcquisitionType.INTEGRATION
            else results[probe_pulse.id][0].shape
        )
    expected_shape = (SWEPT_POINTS,)
    if not average:
        expected_shape = (nshots,) + expected_shape
    if acquisition is not AcquisitionType.DISCRIMINATION:
        expected_shape += (2,)
    assert results_shape == expected_shape


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
    probe_seq = platform.qubits[0].native_gates.MZ.create_sequence()
    pulse = next(iter(probe_seq.values()))[0]
    if parameter is Parameter.amplitude:
        parameter_range = np.random.rand(SWEPT_POINTS)
    else:
        parameter_range = np.random.randint(SWEPT_POINTS, size=SWEPT_POINTS)
    sequence.extend(probe_seq)
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
    results = platform.execute([sequence], options, [[sweeper]])

    assert pulse.id in results
    if average:
        results_shape = (
            results[pulse.id][0].shape
            if acquisition is AcquisitionType.INTEGRATION
            else results[pulse.id][0].shape
        )
    else:
        results_shape = (
            results[pulse.id][0].shape
            if acquisition is AcquisitionType.INTEGRATION
            else results[pulse.id][0].shape
        )

    expected_shape = (SWEPT_POINTS,)
    if not average:
        expected_shape = (nshots,) + expected_shape
    if acquisition is not AcquisitionType.DISCRIMINATION:
        expected_shape += (2,)
    assert results_shape == expected_shape


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
    probe_seq = platform.qubits[0].native_gates.MZ.create_sequence()
    probe_pulse = next(iter(probe_seq.values()))[0]
    sequence[platform.get_qubit(0).drive.name].append(pulse)
    sequence[platform.qubits[0].probe.name].append(Delay(duration=pulse.duration))
    sequence.extend(probe_seq)
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
            platform.qubits[0].probe.name
            if parameter1 is Parameter.frequency
            else platform.qubits[0].flux.name
        )
        sweeper1 = Sweeper(parameter1, parameter_range_1, channels=[channel])
    else:
        sweeper1 = Sweeper(parameter1, parameter_range_1, pulses=[probe_pulse])
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
    results = platform.execute([sequence], options, [[sweeper1], [sweeper2]])

    assert probe_pulse.id in results

    if average:
        results_shape = (
            results[probe_pulse.id][0].shape
            if acquisition is AcquisitionType.INTEGRATION
            else results[probe_pulse.id][0].shape
        )
    else:
        results_shape = (
            results[probe_pulse.id][0].shape
            if acquisition is AcquisitionType.INTEGRATION
            else results[probe_pulse.id][0].shape
        )

    expected_shape = (SWEPT_POINTS, SWEPT_POINTS)
    if not average:
        expected_shape = (nshots,) + expected_shape
    if acquisition is not AcquisitionType.DISCRIMINATION:
        expected_shape += (2,)
    assert results_shape == expected_shape


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
    probe_pulses = {}
    for qubit in platform.qubits:
        probe_seq = platform.qubits[qubit].native_gates.MZ.create_sequence()
        probe_pulses[qubit] = next(iter(probe_seq.values()))[0]
        sequence.extend(probe_seq)
    parameter_range = (
        np.random.rand(SWEPT_POINTS)
        if parameter is Parameter.amplitude
        else np.random.randint(SWEPT_POINTS, size=SWEPT_POINTS)
    )

    if parameter in ChannelParameter:
        sweeper1 = Sweeper(
            parameter,
            parameter_range,
            channels=[qubit.probe.name for qubit in platform.qubits.values()],
        )
    else:
        sweeper1 = Sweeper(
            parameter,
            parameter_range,
            pulses=[probe_pulses[qubit] for qubit in platform.qubits],
        )

    options = ExecutionParameters(
        nshots=nshots,
        averaging_mode=average,
        acquisition_type=acquisition,
    )
    average = not options.averaging_mode is AveragingMode.SINGLESHOT
    results = platform.execute([sequence], options, [[sweeper1]])

    for pulse in probe_pulses.values():
        assert pulse.id in results
        if average:
            results_shape = (
                results[pulse.id][0].shape
                if acquisition is AcquisitionType.INTEGRATION
                else results[pulse.id][0].shape
            )
        else:
            results_shape = (
                results[pulse.id][0].shape
                if acquisition is AcquisitionType.INTEGRATION
                else results[pulse.id][0].shape
            )

        expected_shape = (SWEPT_POINTS,)
        if not average:
            expected_shape = (nshots,) + expected_shape
        if acquisition is not AcquisitionType.DISCRIMINATION:
            expected_shape += (2,)
        assert results_shape == expected_shape


# TODO: add test_dummy_double_sweep_multiplex
