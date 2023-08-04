import numpy as np
import pytest

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters, create_platform
from qibolab.instruments.zhinst import ZhPulse, ZhSweeperLine, Zurich
from qibolab.pulses import (
    IIR,
    SNZ,
    Drag,
    FluxPulse,
    Gaussian,
    Pulse,
    PulseSequence,
    ReadoutPulse,
    Rectangular,
)
from qibolab.sweeper import Parameter, Sweeper

from .conftest import get_instrument


@pytest.mark.parametrize("shape", ["Rectangular", "Gaussian", "GaussianSquare", "Drag", "SNZ", "IIR"])
def test_zhpulse(shape):
    if shape == "Rectangular":
        pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    if shape == "Gaussian":
        pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Gaussian(5), "ch0", qubit=0)
    if shape == "GaussianSquare":
        pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Gaussian(5), "ch0", qubit=0)
    if shape == "Drag":
        pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Drag(5, 0.4), "ch0", qubit=0)
    if shape == "SNZ":
        pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, SNZ(10, 0.4), "ch0", qubit=0)
    if shape == "IIR":
        pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, IIR([10, 1], [0.4, 1], target=Gaussian(5)), "ch0", qubit=0)

    zhpulse = ZhPulse(pulse)
    assert zhpulse.pulse.serial == pulse.serial
    if shape == "SNZ" or shape == "IIR":
        assert len(zhpulse.zhpulse.samples) == 40 / 1e9 * 1e9  # * 2e9 When pulses stop hardcoding SamplingRate
    else:
        assert zhpulse.zhpulse.length == 40e-9


@pytest.mark.parametrize("parameter", [Parameter.bias, Parameter.start])
def test_select_sweeper(dummy_qrc, parameter):
    swept_points = 5
    platform = create_platform("zurich")
    qubits = {0: platform.qubits[0]}
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits.values():
        q = qubit.name
        qd_pulses[q] = platform.create_RX_pulse(q, start=0)
        sequence.add(qd_pulses[q])
        ro_pulses[q] = platform.create_qubit_readout_pulse(q, start=qd_pulses[q].finish)
        sequence.add(ro_pulses[q])

        parameter_range = np.random.randint(swept_points, size=swept_points)
        if parameter is Parameter.start:
            sweeper = Sweeper(parameter, parameter_range, pulses=[qd_pulses[q]])
        if parameter is Parameter.bias:
            sweeper = Sweeper(parameter, parameter_range, qubits=q)

        ZhSweeper = ZhSweeperLine(sweeper, qubit, sequence)
        assert ZhSweeper.sweeper == sweeper


def test_zhinst_setup(dummy_qrc):
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments["EL_ZURO"]
    assert IQM5q.time_of_flight == 280


def test_zhsequence(dummy_qrc):
    qd_pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    ro_pulse = ReadoutPulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch1", qubit=0)
    sequence = PulseSequence()
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)
    IQM5q = create_platform("zurich")
    controller = IQM5q.instruments["EL_ZURO"]

    controller.sequence_zh(sequence, IQM5q.qubits, sweepers=[])
    zhsequence = controller.sequence

    with pytest.raises(AttributeError):
        controller.sequence_zh("sequence", IQM5q.qubits, sweepers=[])
        zhsequence = controller.sequence

    assert len(zhsequence) == 2
    assert len(zhsequence["readout0"]) == 1


def test_zhsequence_multiple_ro(dummy_qrc):
    sequence = PulseSequence()
    qd_pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    sequence.add(qd_pulse)
    ro_pulse = ReadoutPulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch1", qubit=0)
    sequence.add(ro_pulse)
    ro_pulse = ReadoutPulse(0, 5000, 0.05, int(3e9), 0.0, Rectangular(), "ch1", qubit=0)
    sequence.add(ro_pulse)
    platform = create_platform("zurich")

    controller = platform.instruments["EL_ZURO"]
    controller.sequence_zh(sequence, platform.qubits, sweepers=[])
    zhsequence = controller.sequence

    with pytest.raises(AttributeError):
        controller.sequence_zh("sequence", platform.qubits, sweepers=[])
        zhsequence = controller.sequence

    assert len(zhsequence) == 2
    assert len(zhsequence["readout0"]) == 2


def test_zhinst_register_readout_line(dummy_qrc):
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments["EL_ZURO"]
    IQM5q.create_device_setup()
    IQM5q.register_readout_line(platform.qubits[0], intermediate_frequency=int(1e6))

    assert "measure0" in IQM5q.signal_map
    assert "acquire0" in IQM5q.signal_map
    assert "/logical_signal_groups/q0/measure_line" in IQM5q.calibration.calibration_items


def test_zhinst_register_drive_line(dummy_qrc):
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments["EL_ZURO"]
    IQM5q.create_device_setup()
    IQM5q.register_drive_line(platform.qubits[0], intermediate_frequency=int(1e6))

    assert "drive0" in IQM5q.signal_map
    assert "/logical_signal_groups/q0/drive_line" in IQM5q.calibration.calibration_items


def test_zhinst_register_flux_line(dummy_qrc):
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments["EL_ZURO"]
    IQM5q.create_device_setup()
    IQM5q.register_flux_line(platform.qubits[0])

    assert "flux0" in IQM5q.signal_map
    assert "/logical_signal_groups/q0/flux_line" in IQM5q.calibration.calibration_items


def test_experiment_execute_pulse_sequence(dummy_qrc):
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments["EL_ZURO"]
    IQM5q.create_device_setup()

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0], "c0": platform.qubits["c0"]}
    platform.qubits = qubits

    ro_pulses = {}
    qf_pulses = {}
    for qubit in qubits.values():
        q = qubit.name
        qf_pulses[q] = FluxPulse(
            start=0,
            duration=500,
            amplitude=1,
            shape=Rectangular(),
            channel=platform.qubits[q].flux.name,
            qubit=q,
        )
        sequence.add(qf_pulses[q])
        if qubit.flux_coupler:
            continue
        ro_pulses[q] = platform.create_qubit_readout_pulse(q, start=qf_pulses[q].finish)
        sequence.add(ro_pulses[q])

    options = ExecutionParameters(
        relaxation_time=300e-6, acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC
    )

    IQM5q.experiment_flow(qubits, sequence, options)

    assert "flux0" in IQM5q.experiment.signals
    assert "measure0" in IQM5q.experiment.signals
    assert "acquire0" in IQM5q.experiment.signals


def test_experiment_fast_reset_readout(dummy_qrc):
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments["EL_ZURO"]
    IQM5q.create_device_setup()

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0]}
    platform.qubits = qubits

    ro_pulses = {}
    fr_pulses = {}
    for qubit in qubits:
        fr_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    options = ExecutionParameters(
        relaxation_time=300e-6,
        fast_reset=fr_pulses,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    IQM5q.experiment_flow(qubits, sequence, options)

    assert "drive0" in IQM5q.experiment.signals
    assert "measure0" in IQM5q.experiment.signals
    assert "acquire0" in IQM5q.experiment.signals


@pytest.mark.parametrize("fast_reset", [True, False])
def test_experiment_execute_pulse_sequence(dummy_qrc, fast_reset):
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments["EL_ZURO"]
    IQM5q.create_device_setup()

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0]}
    platform.qubits = qubits

    ro_pulses = {}
    qd_pulses = {}
    qf_pulses = {}
    fr_pulses = {}
    for qubit in qubits:
        if fast_reset:
            fr_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.add(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=qd_pulses[qubit].finish)
        sequence.add(ro_pulses[qubit])
        qf_pulses[qubit] = FluxPulse(
            start=0,
            duration=ro_pulses[qubit].se_start,
            amplitude=1,
            shape=Rectangular(),
            channel=platform.qubits[qubit].flux.name,
            qubit=qubit,
        )
        sequence.add(qf_pulses[qubit])

    if fast_reset:
        fast_reset = fr_pulses

    options = ExecutionParameters(
        relaxation_time=300e-6,
        fast_reset=fast_reset,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    IQM5q.experiment_flow(qubits, sequence, options)

    assert "drive0" in IQM5q.experiment.signals
    assert "flux0" in IQM5q.experiment.signals
    assert "measure0" in IQM5q.experiment.signals
    assert "acquire0" in IQM5q.experiment.signals


@pytest.mark.parametrize("parameter1", [Parameter.start, Parameter.duration])
def test_experiment_sweep_single(dummy_qrc, parameter1):
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments["EL_ZURO"]
    IQM5q.create_device_setup()

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0]}

    swept_points = 5
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.add(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=qd_pulses[qubit].finish)
        sequence.add(ro_pulses[qubit])

    parameter_range_1 = (
        np.random.rand(swept_points)
        if parameter1 is Parameter.amplitude
        else np.random.randint(swept_points, size=swept_points)
    )

    sweepers = []
    sweepers.append(Sweeper(parameter1, parameter_range_1, pulses=[qd_pulses[qubit]]))

    options = ExecutionParameters(
        relaxation_time=300e-6, acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC
    )

    IQM5q.sweepers = sweepers

    IQM5q.experiment_flow(qubits, sequence, options, sweepers)

    assert "drive0" in IQM5q.experiment.signals
    assert "measure0" in IQM5q.experiment.signals
    assert "acquire0" in IQM5q.experiment.signals


SweeperParameter = {
    Parameter.frequency,
    Parameter.amplitude,
    Parameter.duration,
    Parameter.start,
    Parameter.relative_phase,
}


@pytest.mark.parametrize("parameter1", Parameter)
@pytest.mark.parametrize("parameter2", Parameter)
def test_experiment_sweep_2d_general(dummy_qrc, parameter1, parameter2):
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments["EL_ZURO"]
    IQM5q.create_device_setup()

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0]}

    swept_points = 5
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.add(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=qd_pulses[qubit].finish)
        sequence.add(ro_pulses[qubit])

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
            sweepers.append(Sweeper(parameter1, parameter_range_1, pulses=[ro_pulses[qubit]]))
    if parameter2 in SweeperParameter:
        if parameter2 is Parameter.amplitude:
            if parameter1 is not Parameter.amplitude:
                sweepers.append(Sweeper(parameter2, parameter_range_2, pulses=[qd_pulses[qubit]]))

    options = ExecutionParameters(
        relaxation_time=300e-6, acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC
    )

    IQM5q.sweepers = sweepers
    rearranging_axes, sweepers = IQM5q.rearrange_sweepers(sweepers)
    IQM5q.experiment_flow(qubits, sequence, options, sweepers)

    assert "drive0" in IQM5q.experiment.signals
    assert "measure0" in IQM5q.experiment.signals
    assert "acquire0" in IQM5q.experiment.signals


def test_experiment_sweep_2d_specific(dummy_qrc):
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments["EL_ZURO"]
    IQM5q.create_device_setup()

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0]}

    swept_points = 5
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.add(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=qd_pulses[qubit].finish)
        sequence.add(ro_pulses[qubit])

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
    sweepers.append(Sweeper(parameter1, parameter_range_1, pulses=[qd_pulses[qubit]]))
    sweepers.append(Sweeper(parameter2, parameter_range_2, pulses=[qd_pulses[qubit]]))

    options = ExecutionParameters(
        relaxation_time=300e-6, acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC
    )

    IQM5q.sweepers = sweepers
    rearranging_axes, sweepers = IQM5q.rearrange_sweepers(sweepers)
    IQM5q.experiment_flow(qubits, sequence, options, sweepers)

    assert "drive0" in IQM5q.experiment.signals
    assert "measure0" in IQM5q.experiment.signals
    assert "acquire0" in IQM5q.experiment.signals
    assert rearranging_axes != [[], []]


@pytest.mark.parametrize("parameter", [Parameter.frequency, Parameter.amplitude, Parameter.bias])
def test_experiment_sweep_punchouts(dummy_qrc, parameter):
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments["EL_ZURO"]
    IQM5q.create_device_setup()

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0]}

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
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

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
    if parameter1 is Parameter.bias:
        sweepers.append(Sweeper(parameter1, parameter_range_1, qubits=[qubits[qubit]]))
    else:
        sweepers.append(Sweeper(parameter1, parameter_range_1, pulses=[ro_pulses[qubit]]))
    sweepers.append(Sweeper(parameter2, parameter_range_2, pulses=[ro_pulses[qubit]]))

    options = ExecutionParameters(
        relaxation_time=300e-6, acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC
    )

    IQM5q.sweepers = sweepers
    rearranging_axes, sweepers = IQM5q.rearrange_sweepers(sweepers)
    IQM5q.experiment_flow(qubits, sequence, options, sweepers)

    assert "measure0" in IQM5q.experiment.signals
    assert "acquire0" in IQM5q.experiment.signals


# TODO: Fix this
def test_sim(dummy_qrc):
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments["EL_ZURO"]
    IQM5q.create_device_setup()
    sequence = PulseSequence()
    qubits = {0: platform.qubits[0]}
    platform.qubits = qubits
    ro_pulses = {}
    qd_pulses = {}
    qf_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.add(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=qd_pulses[qubit].finish)
        sequence.add(ro_pulses[qubit])
        qf_pulses[qubit] = FluxPulse(
            start=0,
            duration=500,
            amplitude=1,
            shape=Rectangular(),
            channel=platform.qubits[qubit].flux.name,
            qubit=qubit,
        )
        sequence.add(qf_pulses[qubit])


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
def test_experiment_execute_pulse_sequence(connected_platform, instrument):
    platform = connected_platform
    platform.setup()

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0], "c0": platform.qubits["c0"]}
    platform.qubits = qubits

    ro_pulses = {}
    qf_pulses = {}
    for qubit in qubits.values():
        q = qubit.name
        qf_pulses[q] = FluxPulse(
            start=0,
            duration=500,
            amplitude=1,
            shape=Rectangular(),
            channel=platform.qubits[q].flux.name,
            qubit=q,
        )
        sequence.add(qf_pulses[q])
        if qubit.flux_coupler:
            continue
        ro_pulses[q] = platform.create_qubit_readout_pulse(q, start=qf_pulses[q].finish)
        sequence.add(ro_pulses[q])

    options = ExecutionParameters(
        relaxation_time=300e-6, acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC
    )

    results = platform.execute_pulse_sequence(
        sequence,
        options,
    )

    assert len(results[ro_pulses[q].serial]) > 0


@pytest.mark.qpu
def test_experiment_sweep_2d_specific(connected_platform, instrument):
    platform = connected_platform
    platform.setup()

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0]}

    swept_points = 5
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.add(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=qd_pulses[qubit].finish)
        sequence.add(ro_pulses[qubit])

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
    sweepers.append(Sweeper(parameter1, parameter_range_1, pulses=[qd_pulses[qubit]]))
    sweepers.append(Sweeper(parameter2, parameter_range_2, pulses=[qd_pulses[qubit]]))

    options = ExecutionParameters(
        relaxation_time=300e-6, acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC
    )

    results = platform.sweep(
        sequence,
        options,
        sweepers[0],
        sweepers[1],
    )

    assert len(results[ro_pulses[qubit].serial]) > 0
