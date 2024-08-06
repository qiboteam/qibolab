import laboneq.dsl.experiment.pulse as laboneq_pulse
import laboneq.simple as lo
import numpy as np
import pytest

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters, create_platform
from qibolab.instruments.zhinst import ProcessedSweeps, Zurich, classify_sweepers
from qibolab.instruments.zhinst.pulse import select_pulse
from qibolab.pulses import (
    Delay,
    Drag,
    Gaussian,
    Iir,
    Pulse,
    PulseSequence,
    PulseType,
    Rectangular,
    Snz,
)
from qibolab.sweeper import Parameter, Sweeper
from qibolab.unrolling import batch

from .conftest import get_instrument


@pytest.mark.parametrize(
    "pulse",
    [
        Pulse(
            duration=40,
            amplitude=0.05,
            frequency=int(3e9),
            relative_phase=0.0,
            envelope=Rectangular(),
            channel="ch0",
            qubit=0,
        ),
        Pulse(
            duration=40,
            amplitude=0.05,
            frequency=int(3e9),
            relative_phase=0.0,
            envelope=Gaussian(rel_sigma=5),
            channel="ch0",
            qubit=0,
        ),
        Pulse(
            duration=40,
            amplitude=0.05,
            frequency=int(3e9),
            relative_phase=0.0,
            envelope=Gaussian(rel_sigma=5),
            channel="ch0",
            qubit=0,
        ),
        Pulse(
            duration=40,
            amplitude=0.05,
            frequency=int(3e9),
            relative_phase=0.0,
            envelope=Drag(rel_sigma=5, beta=0.4),
            channel="ch0",
            qubit=0,
        ),
        Pulse(
            duration=40,
            amplitude=0.05,
            frequency=int(3e9),
            relative_phase=0.0,
            envelope=Snz(t_idling=10, b_amplitude=0.01),
            channel="ch0",
            qubit=0,
        ),
        Pulse(
            duration=40,
            amplitude=0.05,
            frequency=int(3e9),
            relative_phase=0.0,
            envelope=Iir(
                a=np.array([10, 1]), b=np.array([0.4, 1]), target=Gaussian(rel_sigma=5)
            ),
            channel="ch0",
            qubit=0,
        ),
    ],
)
def test_pulse_conversion(pulse):
    shape = pulse.shape
    zhpulse = select_pulse(pulse)
    assert isinstance(zhpulse, laboneq_pulse.Pulse)
    if isinstance(shape, (Snz, Iir)):
        assert len(zhpulse.samples) == 80
    else:
        assert zhpulse.length == 40e-9


def test_classify_sweepers(dummy_qrc):
    platform = create_platform("zurich")
    qubit = platform.qubits[0]
    pulse_1 = Pulse(
        duration=40,
        amplitude=0.05,
        envelope=Gaussian(rel_sigma=5),
        type=PulseType.DRIVE,
    )
    pulse_2 = Pulse(
        duration=40,
        amplitude=0.05,
        envelope=Rectangular(),
        type=PulseType.READOUT,
    )
    amplitude_sweeper = Sweeper(Parameter.amplitude, np.array([1, 2, 3]), [pulse_1])
    readout_amplitude_sweeper = Sweeper(
        Parameter.amplitude, np.array([1, 2, 3, 4, 5]), [pulse_2]
    )
    freq_sweeper = Sweeper(Parameter.frequency, np.array([4, 5, 6, 7]), [pulse_1])
    bias_sweeper = Sweeper(
        Parameter.bias, np.array([3, 2, 1]), channels=[qubit.flux.name]
    )
    nt_sweeps, rt_sweeps = classify_sweepers(
        [amplitude_sweeper, readout_amplitude_sweeper, bias_sweeper, freq_sweeper]
    )

    assert amplitude_sweeper in rt_sweeps
    assert freq_sweeper in rt_sweeps
    assert bias_sweeper in nt_sweeps
    assert readout_amplitude_sweeper in nt_sweeps


def test_processed_sweeps_pulse_properties(dummy_qrc):
    platform = create_platform("zurich")
    zi_instrument = platform.instruments["EL_ZURO"]
    pulse_1 = Pulse(
        duration=40,
        amplitude=0.05,
        envelope=Gaussian(rel_sigma=5),
        type=PulseType.DRIVE,
    )
    pulse_2 = Pulse(
        duration=40,
        amplitude=0.05,
        envelope=Gaussian(rel_sigma=5),
        type=PulseType.DRIVE,
    )
    sweeper_amplitude = Sweeper(
        Parameter.amplitude, np.array([1, 2, 3]), [pulse_1, pulse_2]
    )
    sweeper_duration = Sweeper(Parameter.duration, np.array([1, 2, 3, 4]), [pulse_2])
    processed_sweeps = ProcessedSweeps(
        [sweeper_duration, sweeper_amplitude],
        zi_instrument.channels.values(),
        platform.configs,
    )

    assert len(processed_sweeps.sweeps_for_pulse(pulse_1)) == 1
    assert processed_sweeps.sweeps_for_pulse(pulse_1)[0][0] == Parameter.amplitude
    assert isinstance(
        processed_sweeps.sweeps_for_pulse(pulse_1)[0][1], lo.SweepParameter
    )
    assert len(processed_sweeps.sweeps_for_pulse(pulse_2)) == 2

    assert len(processed_sweeps.sweeps_for_sweeper(sweeper_amplitude)) == 2
    parallel_sweep_ids = {
        s.uid for s in processed_sweeps.sweeps_for_sweeper(sweeper_amplitude)
    }
    assert len(parallel_sweep_ids) == 2
    assert processed_sweeps.sweeps_for_pulse(pulse_1)[0][1].uid in parallel_sweep_ids
    assert any(
        s.uid in parallel_sweep_ids
        for _, s in processed_sweeps.sweeps_for_pulse(pulse_2)
    )

    assert len(processed_sweeps.sweeps_for_sweeper(sweeper_duration)) == 1
    pulse_2_sweep_ids = {s.uid for _, s in processed_sweeps.sweeps_for_pulse(pulse_2)}
    assert len(pulse_2_sweep_ids) == 2
    assert (
        processed_sweeps.sweeps_for_sweeper(sweeper_duration)[0].uid
        in pulse_2_sweep_ids
    )

    assert processed_sweeps.channels_with_sweeps() == set()


# def test_processed_sweeps_readout_amplitude(dummy_qrc):
#     platform = create_platform("zurich")
#     qubit_id, qubit = 0, platform.qubits[0]
#     readout_ch = measure_channel_name(qubit)
#     pulse_readout = Pulse(
#         0,
#         40,
#         0.05,
#         int(3e9),
#         0.0,
#         Rectangular(),
#         readout_ch,
#         PulseType.READOUT,
#         qubit_id,
#     )
#     readout_amplitude_sweeper = Sweeper(
#         Parameter.amplitude, np.array([1, 2, 3, 4]), [pulse_readout]
#     )
#     processed_sweeps = ProcessedSweeps(
#         [readout_amplitude_sweeper], qubits=platform.qubits
#     )
#
#     # Readout amplitude should result into channel property (gain) sweep
#     assert len(processed_sweeps.sweeps_for_pulse(pulse_readout)) == 0
#     assert processed_sweeps.channels_with_sweeps() == {
#         readout_ch,
#     }
#     assert len(processed_sweeps.sweeps_for_channel(readout_ch)) == 1


def test_zhinst_setup(dummy_qrc):
    platform = create_platform("zurich")
    zi_instrument = platform.instruments["EL_ZURO"]
    assert zi_instrument.time_of_flight == 75


def test_zhinst_configure_acquire_line(dummy_qrc):
    platform = create_platform("zurich")
    zi_instrument = platform.instruments["EL_ZURO"]
    qubit = platform.qubits[0]

    zi_instrument.configure_acquire_line(qubit.acquisition.name, platform.configs)

    assert qubit.acquisition.name in zi_instrument.signal_map
    assert (
        "/logical_signal_groups/q0/acquire_line"
        in zi_instrument.calibration.calibration_items
    )


def test_zhinst_configure_iq_line(dummy_qrc):
    platform = create_platform("zurich")
    zi_instrument = platform.instruments["EL_ZURO"]
    qubit = platform.qubits[0]
    zi_instrument.configure_iq_line(qubit.drive.name, platform.configs)
    zi_instrument.configure_iq_line(qubit.probe.name, platform.configs)

    assert qubit.drive.name in zi_instrument.signal_map
    assert (
        "/logical_signal_groups/q0/drive_line"
        in zi_instrument.calibration.calibration_items
    )

    assert qubit.probe.name in zi_instrument.signal_map
    assert (
        "/logical_signal_groups/q0/measure_line"
        in zi_instrument.calibration.calibration_items
    )


def test_zhinst_configure_dc_line(dummy_qrc):
    platform = create_platform("zurich")
    zi_instrument = platform.instruments["EL_ZURO"]
    qubit = platform.qubits[0]
    zi_instrument.configure_dc_line(qubit.flux.name, platform.configs)

    assert qubit.flux.name in zi_instrument.signal_map
    assert (
        "/logical_signal_groups/q0/flux_line"
        in zi_instrument.calibration.calibration_items
    )


def test_experiment_flow(dummy_qrc):
    platform = create_platform("zurich")
    zi_instrument = platform.instruments["EL_ZURO"]

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0], 2: platform.qubits[2]}
    platform.qubits = qubits
    couplers = {}

    for qubit in qubits.values():
        sequence[qubit.flux.name].append(
            Pulse.flux(
                duration=500,
                amplitude=1,
                envelope=Rectangular(),
            )
        )
        sequence[qubit.probe.name].append(Delay(duration=sequence.duration))
        sequence.extend(qubit.native_gates.MZ.create_sequence())

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    zi_instrument.experiment_flow(qubits, couplers, sequence, options)

    assert qubits[0].flux.name in zi_instrument.experiment.signals
    assert qubits[0].probe.name in zi_instrument.experiment.signals
    assert qubits[0].acquisition.name in zi_instrument.experiment.signals


def test_experiment_flow_coupler(dummy_qrc):
    platform = create_platform("zurich")
    zi_instrument = platform.instruments["EL_ZURO"]

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0], 2: platform.qubits[2]}
    platform.qubits = qubits
    couplers = {0: platform.couplers[0]}
    platform.couplers = couplers

    for qubit in qubits.values():
        sequence[qubit.flux.name].append(
            Pulse.flux(
                duration=500,
                amplitude=1,
                envelope=Rectangular(),
            )
        )
        sequence[qubit.probe.name].append(Delay(duration=sequence.duration))
        sequence.extend(qubit.native_gates.MZ.create_sequence())

    for coupler in couplers.values():
        sequence[coupler.flux.name].append(
            Pulse(
                duration=500,
                amplitude=1,
                envelope=Rectangular(),
                type=PulseType.COUPLERFLUX,
            )
        )

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    zi_instrument.experiment_flow(qubits, couplers, sequence, options)

    assert qubits[0].flux.name in zi_instrument.experiment.signals
    assert qubits[0].probe.name in zi_instrument.experiment.signals
    assert qubits[0].acquisition.name in zi_instrument.experiment.signals


def test_sweep_and_play_sim(dummy_qrc):
    """Test end-to-end experiment run using ZI emulated connection."""
    platform = create_platform("zurich")
    zi_instrument = platform.instruments["EL_ZURO"]

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0], 2: platform.qubits[2]}
    platform.qubits = qubits
    couplers = {}

    for qubit in qubits.values():
        sequence[qubit.flux.name].append(
            Pulse.flux(
                duration=500,
                amplitude=1,
                envelope=Rectangular(),
            )
        )
        sequence[qubit.probe.name].append(Delay(duration=sequence.duration))
        sequence.extend(qubit.native_gates.MZ.create_sequence())

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
        nshots=12,
    )

    # check play
    zi_instrument.session = lo.Session(zi_instrument.device_setup)
    zi_instrument.session.connect(do_emulation=True)
    res = zi_instrument.play(platform.configs, [sequence], options, {})
    assert res is not None
    assert all(qubit in res for qubit in qubits)

    # check sweep with empty list of sweeps
    res = zi_instrument.sweep(platform.configs, [sequence], options, {})
    assert res is not None
    assert all(qubit in res for qubit in qubits)

    # check sweep with sweeps
    sweep_1 = Sweeper(
        Parameter.amplitude,
        np.array([1, 2, 3, 4]),
        pulses=[sequence[qubit.flux.name][0] for qubit in qubits.values()],
    )
    sweep_2 = Sweeper(
        Parameter.bias, np.array([1, 2, 3]), channels=[qubits[0].flux.name]
    )
    res = zi_instrument.sweep(
        platform.configs, [sequence], options, {}, sweep_1, sweep_2
    )
    assert res is not None
    assert all(qubit in res for qubit in qubits)


@pytest.mark.parametrize("parameter1", [Parameter.duration])
def test_experiment_sweep_single(dummy_qrc, parameter1):
    platform = create_platform("zurich")
    zi_instrument = platform.instruments["EL_ZURO"]

    qubit_id, qubit = 0, platform.qubits[0]
    couplers = {}

    swept_points = 5
    sequence = PulseSequence()
    sequence.extend(qubit.native_gates.RX.create_sequence(theta=np.pi, phi=0.0))
    sequence[qubit.probe.name].append(Delay(duration=sequence.duration))
    sequence.extend(qubit.native_gates.MZ.create_sequence())

    parameter_range_1 = (
        np.random.rand(swept_points)
        if parameter1 is Parameter.amplitude
        else np.random.randint(swept_points, size=swept_points)
    )

    sweepers = []
    sweepers.append(
        Sweeper(parameter1, parameter_range_1, pulses=[sequence[qubit.drive.name][0]])
    )

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    zi_instrument.experiment_flow({qubit_id: qubit}, couplers, sequence, options)

    assert qubit.drive.name in zi_instrument.experiment.signals
    assert qubit.probe.name in zi_instrument.experiment.signals
    assert qubit.acquisition.name in zi_instrument.experiment.signals


@pytest.mark.parametrize("parameter1", [Parameter.duration])
def test_experiment_sweep_single_coupler(dummy_qrc, parameter1):
    platform = create_platform("zurich")
    zi_instrument = platform.instruments["EL_ZURO"]

    qubits = {0: platform.qubits[0], 2: platform.qubits[2]}
    couplers = {0: platform.couplers[0]}

    swept_points = 5
    sequence = PulseSequence()
    for qubit in qubits.values():
        sequence.extend(qubit.native_gates.RX.create_sequence(theta=np.pi, phi=0.0))
        sequence[qubit.probe.name].append(Delay(duration=sequence.duration))
        sequence.extend(qubit.native_gates.MZ.create_sequence())

    for coupler in couplers.values():
        sequence[coupler.flux.name].append(
            Pulse(
                duration=500,
                amplitude=1,
                envelope=Rectangular(),
                type=PulseType.COUPLERFLUX,
            )
        )

    parameter_range_1 = (
        np.random.rand(swept_points)
        if parameter1 is Parameter.amplitude
        else np.random.randint(swept_points, size=swept_points)
    )

    sweepers = []
    sweepers.append(
        Sweeper(
            parameter1,
            parameter_range_1,
            pulses=[sequence[coupler.flux.name][0] for coupler in couplers.values()],
        )
    )

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    zi_instrument.experiment_flow(qubits, couplers, sequence, options)

    assert couplers[0].flux.name in zi_instrument.experiment.signals
    assert qubits[0].drive.name in zi_instrument.experiment.signals
    assert qubits[0].probe.name in zi_instrument.experiment.signals
    assert qubits[0].acquisition.name in zi_instrument.experiment.signals


SweeperParameter = {
    Parameter.frequency,
    Parameter.amplitude,
    Parameter.duration,
    Parameter.relative_phase,
}


@pytest.mark.parametrize("parameter1", Parameter)
@pytest.mark.parametrize("parameter2", Parameter)
def test_experiment_sweep_2d_general(dummy_qrc, parameter1, parameter2):
    platform = create_platform("zurich")
    zi_instrument = platform.instruments["EL_ZURO"]

    qubits = {0: platform.qubits[0]}
    couplers = {}

    swept_points = 5
    sequence = PulseSequence()
    for qubit in qubits.values():
        sequence.extend(qubit.native_gates.RX.create_sequence(theta=np.pi, phi=0.0))
        sequence[qubit.probe.name].append(Delay(duration=sequence.duration))
        sequence.extend(qubit.native_gates.MZ.create_sequence())

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

    sweepers = []
    if parameter1 in SweeperParameter:
        if parameter1 is not Parameter.start:
            sweepers.append(
                Sweeper(
                    parameter1,
                    parameter_range_1,
                    pulses=[sequence[qubit.probe.name][0] for qubit in qubits.values()],
                )
            )
    if parameter2 in SweeperParameter:
        if parameter2 is Parameter.amplitude:
            if parameter1 is not Parameter.amplitude:
                sweepers.append(
                    Sweeper(
                        parameter2,
                        parameter_range_2,
                        pulses=[
                            sequence[qubit.drive.name][0] for qubit in qubits.values()
                        ],
                    )
                )

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    zi_instrument.experiment_flow(qubits, couplers, sequence, options)

    assert qubits[0].drive.name in zi_instrument.experiment.signals
    assert qubits[0].probe.name in zi_instrument.experiment.signals
    assert qubits[0].acquisition.name in zi_instrument.experiment.signals


def test_experiment_sweep_2d_specific(dummy_qrc):
    platform = create_platform("zurich")
    zi_instrument = platform.instruments["EL_ZURO"]

    qubits = {0: platform.qubits[0]}
    couplers = {}

    swept_points = 5
    sequence = PulseSequence()
    for qubit in qubits.values():
        sequence.extend(qubit.native_gates.RX.create_sequence(theta=np.pi, phi=0.0))
        sequence[qubit.probe.name].append(Delay(duration=sequence.duration))
        sequence.extend(qubit.native_gates.MZ.create_sequence())

    parameter1 = Parameter.relative_phase
    parameter2 = Parameter.frequency

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

    sweepers = []
    qd_pulses = [sequence[qubit.drive.name][0] for qubit in qubits.values()]
    sweepers.append(Sweeper(parameter1, parameter_range_1, pulses=qd_pulses))
    sweepers.append(Sweeper(parameter2, parameter_range_2, pulses=qd_pulses))

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    zi_instrument.experiment_flow(qubits, couplers, sequence, options)

    assert qubits[0].drive.name in zi_instrument.experiment.signals
    assert qubits[0].probe.name in zi_instrument.experiment.signals
    assert qubits[0].acquisition.name in zi_instrument.experiment.signals


@pytest.mark.parametrize(
    "parameter", [Parameter.frequency, Parameter.amplitude, Parameter.bias]
)
def test_experiment_sweep_punchouts(dummy_qrc, parameter):
    platform = create_platform("zurich")
    zi_instrument = platform.instruments["EL_ZURO"]

    qubits = {0: platform.qubits[0]}
    couplers = {}

    if parameter is Parameter.frequency:
        parameter1 = Parameter.frequency
        parameter2 = Parameter.amplitude
    if parameter is Parameter.amplitude:
        parameter1 = Parameter.amplitude
        parameter2 = Parameter.frequency
    if parameter is Parameter.bias:
        parameter1 = Parameter.bias
        parameter2 = Parameter.frequency

    swept_points = 5
    sequence = PulseSequence()
    for qubit in qubits.values():
        sequence.extend(qubit.native_gates.MZ.create_sequence())

    parameter_range_1 = (
        np.random.rand(swept_points)
        if parameter1 in [Parameter.amplitude, Parameter.bias]
        else np.random.randint(swept_points, size=swept_points)
    )

    parameter_range_2 = (
        np.random.rand(swept_points)
        if parameter2 is Parameter.amplitude
        else np.random.randint(swept_points, size=swept_points)
    )

    sweepers = []
    ro_pulses = [sequence[qubit.probe.name][0] for qubit in qubits.values()]
    if parameter1 is Parameter.bias:
        sweepers.append(
            Sweeper(
                parameter1,
                parameter_range_1,
                channels=[qubit.probe.name for qubit in qubits.values()],
            )
        )
    else:
        sweepers.append(Sweeper(parameter1, parameter_range_1, pulses=ro_pulses))
    sweepers.append(Sweeper(parameter2, parameter_range_2, pulses=ro_pulses))

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    zi_instrument.experiment_flow(qubits, couplers, sequence, options)

    assert qubits[0].probe.name in zi_instrument.experiment.signals
    assert qubits[0].acquisition.name in zi_instrument.experiment.signals


def test_batching(dummy_qrc):
    platform = create_platform("zurich")
    instrument = platform.instruments["EL_ZURO"]

    sequence = PulseSequence()
    sequence.extend(
        platform.qubits[0].native_gates.RX.create_sequence(theta=np.pi, phi=0.0)
    )
    sequence.extend(
        platform.qubits[1].native_gates.RX.create_sequence(theta=np.pi, phi=0.0)
    )
    measurement_start = sequence.duration
    sequence[platform.qubits[0].probe.name].append(Delay(duration=measurement_start))
    sequence[platform.qubits[1].probe.name].append(Delay(duration=measurement_start))
    sequence.extend(platform.qubits[0].native_gates.MZ.create_sequence())
    sequence.extend(platform.qubits[1].native_gates.MZ.create_sequence())

    batches = list(batch(600 * [sequence], instrument.bounds))
    # These sequences get limited by the number of measuraments (600/250/2)
    assert len(batches) == 5
    assert len(batches[0]) == 125
    assert len(batches[1]) == 125


@pytest.fixture(scope="module")
def instrument(connected_platform):
    return get_instrument(connected_platform, Zurich)


@pytest.mark.qpu
def test_connections(instrument):
    instrument.start()
    instrument.stop()
    instrument.disconnect()
    instrument.connect()


@pytest.mark.qpu
def test_experiment_execute_qpu(connected_platform, instrument):
    platform = connected_platform
    sequence = PulseSequence()
    qubits = {0: platform.qubits[0], "c0": platform.qubits["c0"]}
    platform.qubits = qubits

    for qubit in qubits.values():
        sequence[qubit.flux.name].append(
            Pulse.flux(
                duration=500,
                amplitude=1,
                envelope=Rectangular(),
            )
        )
        if qubit.flux_coupler:
            continue

        sequence[qubit.probe.name].append(Delay(duration=sequence.duration))
        sequence.extend(qubit.native_gates.MZ.create_sequence())

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    results = platform.execute([sequence], options)
    assert all(len(results[sequence.probe_pulses[q].id]) > 1 for q in qubits)


@pytest.mark.qpu
def test_experiment_sweep_2d_specific_qpu(connected_platform, instrument):
    platform = connected_platform
    qubits = {0: platform.qubits[0]}

    swept_points = 5
    sequence = PulseSequence()
    for qubit in qubits.values():
        sequence.extend(qubit.native_gates.RX.create_sequence(theta=np.pi, phi=0.0))
        sequence[qubit.probe.name].append(Delay(duration=sequence.duration))
        sequence.extend(qubit.native_gates.MZ.create_sequence())

    parameter1 = Parameter.relative_phase
    parameter2 = Parameter.frequency

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

    sweepers = []
    qd_pulses = [sequence[qubit.drive.name][0] for qubit in qubits.values()]
    sweepers.append(Sweeper(parameter1, parameter_range_1, pulses=qd_pulses))
    sweepers.append(Sweeper(parameter2, parameter_range_2, pulses=qd_pulses))

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    results = platform.sweep(
        sequence,
        options,
        sweepers[0],
        sweepers[1],
    )

    assert all(len(results[sequence.probe_pulses[q].id]) for q in qubits) > 0
