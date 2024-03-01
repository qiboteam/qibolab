import math
from collections import defaultdict

import laboneq.dsl.experiment.pulse as laboneq_pulse
import laboneq.simple as lo
import numpy as np
import pytest

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters, create_platform
from qibolab.instruments.zhinst import (
    ProcessedSweeps,
    ZhPulse,
    Zurich,
    acquire_channel_name,
    classify_sweepers,
    measure_channel_name,
)
from qibolab.pulses import (
    IIR,
    SNZ,
    Drag,
    Gaussian,
    Pulse,
    PulseSequence,
    PulseType,
    Rectangular,
)
from qibolab.sweeper import Parameter, Sweeper
from qibolab.unrolling import batch

from .conftest import get_instrument


@pytest.mark.parametrize(
    "pulse",
    [
        Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0),
        Pulse(0, 40, 0.05, int(3e9), 0.0, Gaussian(5), "ch0", qubit=0),
        Pulse(0, 40, 0.05, int(3e9), 0.0, Gaussian(5), "ch0", qubit=0),
        Pulse(0, 40, 0.05, int(3e9), 0.0, Drag(5, 0.4), "ch0", qubit=0),
        Pulse(0, 40, 0.05, int(3e9), 0.0, SNZ(10, 0.01), "ch0", qubit=0),
        Pulse(
            0,
            40,
            0.05,
            int(3e9),
            0.0,
            IIR([10, 1], [0.4, 1], target=Gaussian(5)),
            "ch0",
            qubit=0,
        ),
    ],
)
def test_zhpulse_pulse_conversion(pulse):
    shape = pulse.shape
    zhpulse = ZhPulse(pulse).zhpulse
    assert isinstance(zhpulse, laboneq_pulse.Pulse)
    if isinstance(shape, (SNZ, IIR)):
        assert len(zhpulse.samples) == 80
    else:
        assert zhpulse.length == 40e-9


def test_zhpulse_add_sweeper():
    pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Gaussian(5), "ch", qubit=0)
    zhpulse = ZhPulse(pulse)
    assert zhpulse.zhsweepers == []
    assert zhpulse.delay_sweeper is None

    zhpulse.add_sweeper(
        Parameter.duration, lo.SweepParameter(values=np.array([1, 2, 3]))
    )
    assert len(zhpulse.zhsweepers) == 1
    assert zhpulse.delay_sweeper is None

    zhpulse.add_sweeper(
        Parameter.start, lo.SweepParameter(values=np.array([4, 5, 6, 7]))
    )
    assert len(zhpulse.zhsweepers) == 1
    assert zhpulse.delay_sweeper is not None

    zhpulse.add_sweeper(
        Parameter.amplitude, lo.SweepParameter(values=np.array([3, 2, 1, 0]))
    )
    assert len(zhpulse.zhsweepers) == 2
    assert zhpulse.delay_sweeper is not None


def test_measure_channel_name(dummy_qrc):
    platform = create_platform("zurich")
    qubits = platform.qubits.values()
    meas_ch_names = {measure_channel_name(q) for q in qubits}
    assert len(qubits) > 0
    assert len(meas_ch_names) == len(qubits)


def test_acquire_channel_name(dummy_qrc):
    platform = create_platform("zurich")
    qubits = platform.qubits.values()
    acq_ch_names = {acquire_channel_name(q) for q in qubits}
    assert len(qubits) > 0
    assert len(acq_ch_names) == len(qubits)


def test_classify_sweepers(dummy_qrc):
    platform = create_platform("zurich")
    qubit_id, qubit = 0, platform.qubits[0]
    pulse_1 = Pulse(0, 40, 0.05, int(3e9), 0.0, Gaussian(5), "ch0", qubit=qubit_id)
    pulse_2 = Pulse(
        0,
        40,
        0.05,
        int(3e9),
        0.0,
        Rectangular(),
        "ch7",
        PulseType.READOUT,
        qubit=qubit_id,
    )
    amplitude_sweeper = Sweeper(Parameter.amplitude, np.array([1, 2, 3]), [pulse_1])
    readout_amplitude_sweeper = Sweeper(
        Parameter.amplitude, np.array([1, 2, 3, 4, 5]), [pulse_2]
    )
    freq_sweeper = Sweeper(Parameter.frequency, np.array([4, 5, 6, 7]), [pulse_1])
    bias_sweeper = Sweeper(Parameter.bias, np.array([3, 2, 1]), qubits=[qubit])
    nt_sweeps, rt_sweeps = classify_sweepers(
        [amplitude_sweeper, readout_amplitude_sweeper, bias_sweeper, freq_sweeper]
    )

    assert amplitude_sweeper in rt_sweeps
    assert freq_sweeper in rt_sweeps
    assert bias_sweeper in nt_sweeps
    assert readout_amplitude_sweeper in nt_sweeps


def test_processed_sweeps_pulse_properties(dummy_qrc):
    platform = create_platform("zurich")
    qubit_id_1, qubit_1 = 0, platform.qubits[0]
    qubit_id_2, qubit_2 = 3, platform.qubits[3]
    pulse_1 = Pulse(
        0, 40, 0.05, int(3e9), 0.0, Gaussian(5), qubit_1.drive.name, qubit=qubit_id_1
    )
    pulse_2 = Pulse(
        0, 40, 0.05, int(3e9), 0.0, Gaussian(5), qubit_2.drive.name, qubit=qubit_id_2
    )
    sweeper_amplitude = Sweeper(
        Parameter.amplitude, np.array([1, 2, 3]), [pulse_1, pulse_2]
    )
    sweeper_duration = Sweeper(Parameter.duration, np.array([1, 2, 3, 4]), [pulse_2])
    processed_sweeps = ProcessedSweeps(
        [sweeper_duration, sweeper_amplitude], qubits=platform.qubits
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


def test_processed_sweeps_frequency(dummy_qrc):
    platform = create_platform("zurich")
    qubit_id, qubit = 1, platform.qubits[1]
    pulse = Pulse(
        0, 40, 0.05, int(3e9), 0.0, Gaussian(5), qubit.drive.name, qubit=qubit_id
    )
    freq_sweeper = Sweeper(Parameter.frequency, np.array([1, 2, 3]), [pulse])
    processed_sweeps = ProcessedSweeps([freq_sweeper], platform.qubits)

    # Frequency sweepers should result into channel property sweeps
    assert len(processed_sweeps.sweeps_for_pulse(pulse)) == 0
    assert processed_sweeps.channels_with_sweeps() == {qubit.drive.name}
    assert len(processed_sweeps.sweeps_for_channel(qubit.drive.name)) == 1

    with pytest.raises(ValueError):
        flux_pulse = Pulse(
            0,
            40,
            0.05,
            int(3e9),
            0.0,
            Gaussian(5),
            qubit.flux.name,
            PulseType.FLUX,
            qubit=qubit_id,
        )
        freq_sweeper = Sweeper(
            Parameter.frequency, np.array([1, 3, 5, 7]), [flux_pulse]
        )
        ProcessedSweeps([freq_sweeper], platform.qubits)


def test_processed_sweeps_readout_amplitude(dummy_qrc):
    platform = create_platform("zurich")
    qubit_id, qubit = 0, platform.qubits[0]
    readout_ch = measure_channel_name(qubit)
    pulse_readout = Pulse(
        0,
        40,
        0.05,
        int(3e9),
        0.0,
        Rectangular(),
        readout_ch,
        PulseType.READOUT,
        qubit_id,
    )
    readout_amplitude_sweeper = Sweeper(
        Parameter.amplitude, np.array([1, 2, 3, 4]), [pulse_readout]
    )
    processed_sweeps = ProcessedSweeps(
        [readout_amplitude_sweeper], qubits=platform.qubits
    )

    # Readout amplitude should result into channel property (gain) sweep
    assert len(processed_sweeps.sweeps_for_pulse(pulse_readout)) == 0
    assert processed_sweeps.channels_with_sweeps() == {
        readout_ch,
    }
    assert len(processed_sweeps.sweeps_for_channel(readout_ch)) == 1


def test_zhinst_setup(dummy_qrc):
    platform = create_platform("zurich")
    IQM5q = platform.instruments["EL_ZURO"]
    assert IQM5q.time_of_flight == 75


def test_zhsequence(dummy_qrc):
    IQM5q = create_platform("zurich")
    controller = IQM5q.instruments["EL_ZURO"]

    drive_channel, readout_channel = IQM5q.qubits[0].drive.name, measure_channel_name(
        IQM5q.qubits[0]
    )
    qd_pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), drive_channel, qubit=0)
    ro_pulse = Pulse(
        0,
        40,
        0.05,
        int(3e9),
        0.0,
        Rectangular(),
        readout_channel,
        PulseType.READOUT,
        qubit=0,
    )
    sequence = PulseSequence()
    sequence.append(qd_pulse)
    sequence.append(qd_pulse)
    sequence.append(ro_pulse)

    zhsequence = controller.sequence_zh(sequence, IQM5q.qubits)

    assert len(zhsequence) == 2
    assert len(zhsequence[drive_channel]) == 2
    assert len(zhsequence[readout_channel]) == 1

    with pytest.raises(AttributeError):
        controller.sequence_zh("sequence", IQM5q.qubits)


def test_zhsequence_couplers(dummy_qrc):
    IQM5q = create_platform("zurich")
    controller = IQM5q.instruments["EL_ZURO"]

    drive_channel, readout_channel = IQM5q.qubits[0].drive.name, measure_channel_name(
        IQM5q.qubits[0]
    )
    couplerflux_channel = IQM5q.couplers[0].flux.name
    qd_pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), drive_channel, qubit=0)
    ro_pulse = Pulse(
        0,
        40,
        0.05,
        int(3e9),
        0.0,
        Rectangular(),
        readout_channel,
        PulseType.READOUT,
        qubit=0,
    )
    qc_pulse = Pulse.flux(
        0, 40, 0.05, Rectangular(), channel=couplerflux_channel, qubit=3
    )
    qc_pulse.type = PulseType.COUPLERFLUX
    sequence = PulseSequence()
    sequence.append(qd_pulse)
    sequence.append(ro_pulse)
    sequence.append(qc_pulse)

    zhsequence = controller.sequence_zh(sequence, IQM5q.qubits)

    assert len(zhsequence) == 3
    assert len(zhsequence[couplerflux_channel]) == 1


def test_zhsequence_multiple_ro(dummy_qrc):
    platform = create_platform("zurich")
    readout_channel = measure_channel_name(platform.qubits[0])
    sequence = PulseSequence()
    qd_pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    sequence.append(qd_pulse)
    ro_pulse = Pulse(
        0,
        40,
        0.05,
        int(3e9),
        0.0,
        Rectangular(),
        readout_channel,
        PulseType.READOUT,
        qubit=0,
    )
    sequence.append(ro_pulse)
    ro_pulse = Pulse(
        0,
        5000,
        0.05,
        int(3e9),
        0.0,
        Rectangular(),
        readout_channel,
        PulseType.READOUT,
        qubit=0,
    )
    sequence.append(ro_pulse)
    platform = create_platform("zurich")

    controller = platform.instruments["EL_ZURO"]
    zhsequence = controller.sequence_zh(sequence, platform.qubits)

    assert len(zhsequence) == 2
    assert len(zhsequence[readout_channel]) == 2


def test_zhinst_register_readout_line(dummy_qrc):
    platform = create_platform("zurich")
    IQM5q = platform.instruments["EL_ZURO"]
    qubit = platform.qubits[0]

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    IQM5q.register_readout_line(qubit, intermediate_frequency=int(1e6), options=options)

    assert measure_channel_name(qubit) in IQM5q.signal_map
    assert acquire_channel_name(qubit) in IQM5q.signal_map
    assert (
        "/logical_signal_groups/q0/measure_line" in IQM5q.calibration.calibration_items
    )


def test_zhinst_register_drive_line(dummy_qrc):
    platform = create_platform("zurich")
    IQM5q = platform.instruments["EL_ZURO"]
    qubit = platform.qubits[0]
    IQM5q.register_drive_line(qubit, intermediate_frequency=int(1e6))

    assert qubit.drive.name in IQM5q.signal_map
    assert "/logical_signal_groups/q0/drive_line" in IQM5q.calibration.calibration_items


def test_zhinst_register_flux_line(dummy_qrc):
    platform = create_platform("zurich")
    IQM5q = platform.instruments["EL_ZURO"]
    qubit = platform.qubits[0]
    IQM5q.register_flux_line(qubit)

    assert qubit.flux.name in IQM5q.signal_map
    assert "/logical_signal_groups/q0/flux_line" in IQM5q.calibration.calibration_items


def test_experiment_flow(dummy_qrc):
    platform = create_platform("zurich")
    IQM5q = platform.instruments["EL_ZURO"]

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0], 2: platform.qubits[2]}
    platform.qubits = qubits
    couplers = {}

    ro_pulses = {}
    qf_pulses = {}
    for qubit in qubits.values():
        q = qubit.name
        qf_pulses[q] = Pulse.flux(
            start=0,
            duration=500,
            amplitude=1,
            shape=Rectangular(),
            channel=platform.qubits[q].flux.name,
            qubit=q,
        )
        sequence.append(qf_pulses[q])
        ro_pulses[q] = platform.create_qubit_readout_pulse(q, start=qf_pulses[q].finish)
        sequence.append(ro_pulses[q])

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    IQM5q.experiment_flow(qubits, couplers, sequence, options)

    assert qubits[0].flux.name in IQM5q.experiment.signals
    assert measure_channel_name(qubits[0]) in IQM5q.experiment.signals
    assert acquire_channel_name(qubits[0]) in IQM5q.experiment.signals


def test_experiment_flow_coupler(dummy_qrc):
    platform = create_platform("zurich")
    IQM5q = platform.instruments["EL_ZURO"]

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0], 2: platform.qubits[2]}
    platform.qubits = qubits
    couplers = {0: platform.couplers[0]}
    platform.couplers = couplers

    ro_pulses = {}
    qf_pulses = {}
    for qubit in qubits.values():
        q = qubit.name
        qf_pulses[q] = Pulse.flux(
            start=0,
            duration=500,
            amplitude=1,
            shape=Rectangular(),
            channel=platform.qubits[q].flux.name,
            qubit=q,
        )
        sequence.append(qf_pulses[q])
        ro_pulses[q] = platform.create_qubit_readout_pulse(q, start=qf_pulses[q].finish)
        sequence.append(ro_pulses[q])

    cf_pulses = {}
    for coupler in couplers.values():
        c = coupler.name
        cf_pulses[c] = Pulse.flux(
            start=0,
            duration=500,
            amplitude=1,
            shape=Rectangular(),
            channel=platform.couplers[c].flux.name,
            qubit=c,
        )
        cf_pulses[c].type = PulseType.COUPLERFLUX
        sequence.append(cf_pulses[c])

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    IQM5q.experiment_flow(qubits, couplers, sequence, options)

    assert qubits[0].flux.name in IQM5q.experiment.signals
    assert measure_channel_name(qubits[0]) in IQM5q.experiment.signals
    assert acquire_channel_name(qubits[0]) in IQM5q.experiment.signals


def test_sweep_and_play_sim(dummy_qrc):
    """Test end-to-end experiment run using ZI emulated connection."""
    platform = create_platform("zurich")
    IQM5q = platform.instruments["EL_ZURO"]

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0], 2: platform.qubits[2]}
    platform.qubits = qubits
    couplers = {}

    ro_pulses = {}
    qf_pulses = {}
    for qubit in qubits.values():
        q = qubit.name
        qf_pulses[q] = Pulse.flux(
            start=0,
            duration=500,
            amplitude=1,
            shape=Rectangular(),
            channel=platform.qubits[q].flux.name,
            qubit=q,
        )
        sequence.append(qf_pulses[q])
        ro_pulses[q] = platform.create_qubit_readout_pulse(q, start=qf_pulses[q].finish)
        sequence.append(ro_pulses[q])

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
        nshots=12,
    )

    # check play
    IQM5q.session = lo.Session(IQM5q.device_setup)
    IQM5q.session.connect(do_emulation=True)
    res = IQM5q.play(qubits, couplers, sequence, options)
    assert res is not None
    assert all(qubit in res for qubit in qubits)

    # check sweep with empty list of sweeps
    res = IQM5q.sweep(qubits, couplers, sequence, options)
    assert res is not None
    assert all(qubit in res for qubit in qubits)

    # check sweep with sweeps
    sweep_1 = Sweeper(Parameter.start, np.array([1, 2, 3, 4]), list(qf_pulses.values()))
    sweep_2 = Sweeper(Parameter.bias, np.array([1, 2, 3]), qubits=[qubits[0]])
    res = IQM5q.sweep(qubits, couplers, sequence, options, sweep_1, sweep_2)
    assert res is not None
    assert all(qubit in res for qubit in qubits)


@pytest.mark.parametrize("parameter1", [Parameter.duration])
def test_experiment_sweep_single(dummy_qrc, parameter1):
    platform = create_platform("zurich")
    IQM5q = platform.instruments["EL_ZURO"]

    qubits = {0: platform.qubits[0]}
    couplers = {}

    swept_points = 5
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.append(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.append(ro_pulses[qubit])

    parameter_range_1 = (
        np.random.rand(swept_points)
        if parameter1 is Parameter.amplitude
        else np.random.randint(swept_points, size=swept_points)
    )

    sweepers = []
    sweepers.append(Sweeper(parameter1, parameter_range_1, pulses=[qd_pulses[qubit]]))

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    IQM5q.experiment_flow(qubits, couplers, sequence, options)

    assert qubits[0].drive.name in IQM5q.experiment.signals
    assert measure_channel_name(qubits[0]) in IQM5q.experiment.signals
    assert acquire_channel_name(qubits[0]) in IQM5q.experiment.signals


@pytest.mark.parametrize("parameter1", [Parameter.duration])
def test_experiment_sweep_single_coupler(dummy_qrc, parameter1):
    platform = create_platform("zurich")
    IQM5q = platform.instruments["EL_ZURO"]

    qubits = {0: platform.qubits[0], 2: platform.qubits[2]}
    couplers = {0: platform.couplers[0]}

    swept_points = 5
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.append(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.append(ro_pulses[qubit])

    cf_pulses = {}
    for coupler in couplers.values():
        c = coupler.name
        cf_pulses[c] = Pulse.flux(
            start=0,
            duration=500,
            amplitude=1,
            shape=Rectangular(),
            channel=platform.couplers[c].flux.name,
            qubit=c,
        )
        cf_pulses[c].type = PulseType.COUPLERFLUX
        sequence.append(cf_pulses[c])

    parameter_range_1 = (
        np.random.rand(swept_points)
        if parameter1 is Parameter.amplitude
        else np.random.randint(swept_points, size=swept_points)
    )

    sweepers = []
    sweepers.append(Sweeper(parameter1, parameter_range_1, pulses=[cf_pulses[c]]))

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    IQM5q.experiment_flow(qubits, couplers, sequence, options)

    assert couplers[0].flux.name in IQM5q.experiment.signals
    assert qubits[0].drive.name in IQM5q.experiment.signals
    assert measure_channel_name(qubits[0]) in IQM5q.experiment.signals
    assert acquire_channel_name(qubits[0]) in IQM5q.experiment.signals


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
    IQM5q = platform.instruments["EL_ZURO"]

    qubits = {0: platform.qubits[0]}
    couplers = {}

    swept_points = 5
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.append(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.append(ro_pulses[qubit])

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
                Sweeper(parameter1, parameter_range_1, pulses=[ro_pulses[qubit]])
            )
    if parameter2 in SweeperParameter:
        if parameter2 is Parameter.amplitude:
            if parameter1 is not Parameter.amplitude:
                sweepers.append(
                    Sweeper(parameter2, parameter_range_2, pulses=[qd_pulses[qubit]])
                )

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    IQM5q.experiment_flow(qubits, couplers, sequence, options)

    assert qubits[0].drive.name in IQM5q.experiment.signals
    assert measure_channel_name(qubits[0]) in IQM5q.experiment.signals
    assert acquire_channel_name(qubits[0]) in IQM5q.experiment.signals


def test_experiment_sweep_2d_specific(dummy_qrc):
    platform = create_platform("zurich")
    IQM5q = platform.instruments["EL_ZURO"]

    qubits = {0: platform.qubits[0]}
    couplers = {}

    swept_points = 5
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.append(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.append(ro_pulses[qubit])

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
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    IQM5q.experiment_flow(qubits, couplers, sequence, options)

    assert qubits[0].drive.name in IQM5q.experiment.signals
    assert measure_channel_name(qubits[0]) in IQM5q.experiment.signals
    assert acquire_channel_name(qubits[0]) in IQM5q.experiment.signals


@pytest.mark.parametrize(
    "parameter", [Parameter.frequency, Parameter.amplitude, Parameter.bias]
)
def test_experiment_sweep_punchouts(dummy_qrc, parameter):
    platform = create_platform("zurich")
    IQM5q = platform.instruments["EL_ZURO"]

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
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.append(ro_pulses[qubit])

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
    if parameter1 is Parameter.bias:
        sweepers.append(Sweeper(parameter1, parameter_range_1, qubits=[qubits[qubit]]))
    else:
        sweepers.append(
            Sweeper(parameter1, parameter_range_1, pulses=[ro_pulses[qubit]])
        )
    sweepers.append(Sweeper(parameter2, parameter_range_2, pulses=[ro_pulses[qubit]]))

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    IQM5q.experiment_flow(qubits, couplers, sequence, options)

    assert measure_channel_name(qubits[0]) in IQM5q.experiment.signals
    assert acquire_channel_name(qubits[0]) in IQM5q.experiment.signals


def test_batching(dummy_qrc):
    platform = create_platform("zurich")
    instrument = platform.instruments["EL_ZURO"]

    sequence = PulseSequence()
    sequence.append(platform.create_RX_pulse(0, start=0))
    sequence.append(platform.create_RX_pulse(1, start=0))
    measurement_start = sequence.finish
    sequence.append(platform.create_MZ_pulse(0, start=measurement_start))
    sequence.append(platform.create_MZ_pulse(1, start=measurement_start))

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
def test_experiment_execute_pulse_sequence_qpu(connected_platform, instrument):
    platform = connected_platform
    sequence = PulseSequence()
    qubits = {0: platform.qubits[0], "c0": platform.qubits["c0"]}
    platform.qubits = qubits

    ro_pulses = {}
    qf_pulses = {}
    for qubit in qubits.values():
        q = qubit.name
        qf_pulses[q] = Pulse.flux(
            start=0,
            duration=500,
            amplitude=1,
            shape=Rectangular(),
            channel=platform.qubits[q].flux.name,
            qubit=q,
        )
        sequence.append(qf_pulses[q])
        if qubit.flux_coupler:
            continue
        ro_pulses[q] = platform.create_qubit_readout_pulse(q, start=qf_pulses[q].finish)
        sequence.append(ro_pulses[q])

    options = ExecutionParameters(
        relaxation_time=300e-6,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    results = platform.execute_pulse_sequence(
        sequence,
        options,
    )

    assert len(results[ro_pulses[q].id]) > 0


@pytest.mark.qpu
def test_experiment_sweep_2d_specific_qpu(connected_platform, instrument):
    platform = connected_platform
    qubits = {0: platform.qubits[0]}

    swept_points = 5
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.append(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.append(ro_pulses[qubit])

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

    assert len(results[ro_pulses[qubit].id]) > 0


def get_previous_subsequence_finish(instrument, name):
    """Look recursively for sub_section finish times."""
    section = next(
        iter(ch for ch in instrument.experiment.sections[0].children if ch.uid == name)
    )
    finish = defaultdict(int)
    for pulse in section.children:
        try:
            finish[pulse.signal] += pulse.time
        except AttributeError:
            # not a laboneq Delay class object, skipping
            pass
        try:
            finish[pulse.signal] += pulse.pulse.length
        except AttributeError:
            # not a laboneq PlayPulse class object, skipping
            pass
    return max(finish.values())


def test_experiment_measurement_sequence(dummy_qrc):
    platform = create_platform("zurich")
    IQM5q = platform.instruments["EL_ZURO"]

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0]}
    platform.qubits = qubits
    couplers = {}

    readout_pulse_start = 40

    for qubit in qubits:
        qubit_drive_pulse_1 = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=40
        )
        ro_pulse = platform.create_qubit_readout_pulse(qubit, start=readout_pulse_start)
        qubit_drive_pulse_2 = platform.create_qubit_drive_pulse(
            qubit, start=readout_pulse_start + 50, duration=40
        )
        sequence.append(qubit_drive_pulse_1)
        sequence.append(ro_pulse)
        sequence.append(qubit_drive_pulse_2)

    options = ExecutionParameters(
        relaxation_time=4,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    IQM5q.experiment_flow(qubits, couplers, sequence, options)
    measure_start = 0
    for section in IQM5q.experiment.sections[0].children:
        if section.uid == "measure_0":
            measure_start += get_previous_subsequence_finish(IQM5q, section.play_after)
            for pulse in section.children:
                try:
                    if pulse.signal == measure_channel_name(qubits[0]):
                        measure_start += pulse.time
                except AttributeError:
                    # not a laboneq delay class object, skipping
                    pass

    assert math.isclose(measure_start * 1e9, readout_pulse_start, rel_tol=1e-4)
