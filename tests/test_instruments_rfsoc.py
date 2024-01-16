"""Tests for RFSoC driver."""

from dataclasses import asdict

import numpy as np
import pytest
import qibosoq.components.base as rfsoc
import qibosoq.components.pulses as rfsoc_pulses

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters, create_platform
from qibolab.instruments.rfsoc import RFSoC
from qibolab.instruments.rfsoc.convert import (
    convert,
    convert_units_sweeper,
    replace_pulse_shape,
)
from qibolab.platform import Qubit
from qibolab.pulses import Drag, Gaussian, Pulse, PulseSequence, PulseType, Rectangular
from qibolab.result import (
    AveragedIntegratedResults,
    AveragedSampleResults,
    IntegratedResults,
)
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from .conftest import get_instrument


def test_convert_default(dummy_qrc):
    """Test convert function raises errors when parameter have wrong types."""
    platform = create_platform("rfsoc")
    integer = 12
    qubits = platform.qubits
    sequence = PulseSequence()
    sequence.append(Pulse(0, 40, 0.9, 50e6, 0, Drag(5, 2), 0, PulseType.DRIVE, 0))
    parameter = Parameter.frequency

    with pytest.raises(ValueError):
        res = convert(integer)  # this conversion does not exist

    with pytest.raises(ValueError):
        res = convert(qubits, sequence)  # the order is wrong

    with pytest.raises(TypeError):
        # functools understand that is a convert_parameter and raises an error for the int
        _ = convert(parameter, integer)


def test_convert_qubit(dummy_qrc):
    """Tests conversion from `qibolab.platforms.abstract.Qubit` to
    `rfsoc.Qubit`.

    Test conversion for flux qubit and for non-flux qubit.
    """
    platform = create_platform("rfsoc")
    qubit = platform.qubits[0]
    qubit.flux.port = platform.instruments["tii_rfsoc4x2"].ports(4)
    qubit.flux.offset = 0.05
    qubit = convert(qubit)
    targ = rfsoc.Qubit(0.05, 4)

    assert qubit == targ

    platform = create_platform("rfsoc")
    qubit = platform.qubits[0]
    qubit.flux = None
    qubit = convert(qubit)
    targ = rfsoc.Qubit(0.0, None)

    assert qubit == targ


def test_replace_pulse_shape(dummy_qrc):
    """Test rfsoc pulse conversions."""

    pulse = rfsoc_pulses.Pulse(50, 0.9, 0, 0, 0.04, "name", "drive", 4, None)

    new_pulse = replace_pulse_shape(pulse, Rectangular(), sampling_rate=1)
    assert isinstance(new_pulse, rfsoc_pulses.Rectangular)
    for key in asdict(pulse):
        assert asdict(pulse)[key] == asdict(new_pulse)[key]

    new_pulse = replace_pulse_shape(pulse, Gaussian(5), sampling_rate=1)
    assert isinstance(new_pulse, rfsoc_pulses.Gaussian)
    assert new_pulse.rel_sigma == 5
    for key in asdict(pulse):
        assert asdict(pulse)[key] == asdict(new_pulse)[key]

    new_pulse = replace_pulse_shape(pulse, Drag(5, 7), sampling_rate=1)
    assert isinstance(new_pulse, rfsoc_pulses.Drag)
    assert new_pulse.rel_sigma == 5
    assert new_pulse.beta == 7
    for key in asdict(pulse):
        assert asdict(pulse)[key] == asdict(new_pulse)[key]


def test_convert_pulse(dummy_qrc):
    """Tests conversion from `qibolab.pulses.Pulse` to `rfsoc.Pulse`.

    Test drive pulse (gaussian and drag), and readout with LO.
    """
    platform = create_platform("rfsoc")
    controller = platform.instruments["tii_rfsoc4x2"]
    qubit = platform.qubits[0]
    qubit.drive.port = controller.ports(4)
    qubit.readout.port = controller.ports(2)
    qubit.feedback.port = controller.ports(1)
    qubit.readout.local_oscillator.frequency = 1e6

    pulse = Pulse(
        start=0,
        duration=40,
        amplitude=0.9,
        frequency=50e6,
        relative_phase=0,
        shape=Drag(5, 2),
        channel=0,
        type=PulseType.DRIVE,
        qubit=0,
    )
    targ = rfsoc_pulses.Drag(
        type="drive",
        frequency=50,
        amplitude=0.9,
        start_delay=0,
        duration=0.04,
        adc=None,
        dac=4,
        name=pulse.serial,
        relative_phase=0,
        rel_sigma=5,
        beta=2,
    )
    assert convert(pulse, platform.qubits, 0, sampling_rate=1) == targ

    pulse = Pulse(
        start=0,
        duration=40,
        amplitude=0.9,
        frequency=50e6,
        relative_phase=0,
        shape=Gaussian(2),
        channel=0,
        type=PulseType.DRIVE,
        qubit=0,
    )
    targ = rfsoc_pulses.Gaussian(
        frequency=50,
        amplitude=0.9,
        start_delay=0,
        relative_phase=0,
        duration=0.04,
        name=pulse.serial,
        type="drive",
        dac=4,
        adc=None,
        rel_sigma=2,
    )
    assert convert(pulse, platform.qubits, 0, sampling_rate=1) == targ

    pulse = Pulse(
        start=0,
        duration=40,
        amplitude=0.9,
        frequency=50e6,
        relative_phase=0,
        shape=Rectangular(),
        channel=0,
        type=PulseType.READOUT,
        qubit=0,
    )
    targ = rfsoc_pulses.Rectangular(
        frequency=49,
        amplitude=0.9,
        start_delay=0,
        relative_phase=0,
        duration=0.04,
        name=pulse.serial,
        type="readout",
        dac=2,
        adc=1,
    )
    assert convert(pulse, platform.qubits, 0, sampling_rate=1) == targ


def test_convert_units_sweeper(dummy_qrc):
    """Tests units conversion for `rfsoc.Sweeper` objects.

    Test frequency conversion (with and without LO), start and relative
    phase sweepers.
    """
    platform = create_platform("rfsoc")
    qubit = platform.qubits[0]
    qubit.drive.ports = [("name", 4)]
    qubit.readout.ports = [("name", 2)]
    qubit.feedback.ports = [("name", 1)]
    qubit.readout.local_oscillator.frequency = 1e6

    seq = PulseSequence()
    pulse0 = Pulse(0, 40, 0.9, 50e6, 0, Gaussian(2), 0, PulseType.DRIVE, 0)
    pulse1 = Pulse(40, 40, 0.9, 50e6, 0, Rectangular(), 0, PulseType.READOUT, 0)
    seq.append(pulse0)
    seq.append(pulse1)

    # frequency sweeper
    sweeper = rfsoc.Sweeper(
        parameters=[rfsoc.Parameter.FREQUENCY],
        indexes=[1],
        starts=[0],
        stops=[10e6],
        expts=100,
    )
    convert_units_sweeper(sweeper, seq, platform.qubits)

    assert sweeper.starts == [-1]
    assert sweeper.stops == [9]

    qubit.readout.local_oscillator.frequency = 0
    sweeper = rfsoc.Sweeper(
        parameters=[rfsoc.Parameter.FREQUENCY],
        indexes=[1],
        starts=[0],
        stops=[10e6],
        expts=100,
    )
    convert_units_sweeper(sweeper, seq, platform.qubits)
    assert sweeper.starts == [0]
    assert sweeper.stops == [10]

    # start sweeper
    sweeper = rfsoc.Sweeper(
        parameters=[rfsoc.Parameter.DELAY, rfsoc.Parameter.DELAY],
        indexes=[0, 1],
        starts=[0, 40],
        stops=[100, 140],
        expts=100,
    )
    convert_units_sweeper(sweeper, seq, platform.qubits)
    assert (sweeper.starts == [0, 0.04]).all()
    assert (sweeper.stops == [0.1, 0.14]).all()

    # phase sweeper
    sweeper = rfsoc.Sweeper(
        parameters=[rfsoc.Parameter.RELATIVE_PHASE],
        indexes=[0],
        starts=[0],
        stops=[np.pi],
        expts=180,
    )
    convert_units_sweeper(sweeper, seq, platform.qubits)
    assert sweeper.starts == [0]
    assert sweeper.stops == [180]


def test_convert_sweep(dummy_qrc):
    """Test conversion between `Sweeper` and `rfsoc.Sweeper` objects.

    Test bias sweep, amplitude error, frequency sweep, duration, start.
    """
    platform = create_platform("rfsoc")
    qubit = platform.qubits[0]
    qubit.flux.offset = 0.05
    qubit.flux.ports = [("name", 4)]
    qubit.drive.ports = [("name", 4)]
    qubit.readout.ports = [("name", 2)]

    seq = PulseSequence()
    pulse0 = Pulse(0, 40, 0.9, 50e6, 0, Gaussian(2), 0, PulseType.DRIVE, 0)
    pulse1 = Pulse(40, 40, 0.9, 50e6, 0, Rectangular(), 0, PulseType.READOUT, 0)
    seq.append(pulse0)
    seq.append(pulse1)

    sweeper = Sweeper(
        parameter=Parameter.bias, values=np.arange(-0.5, +0.5, 0.1), qubits=[qubit]
    )
    rfsoc_sweeper = convert(sweeper, seq, platform.qubits)
    targ = rfsoc.Sweeper(
        expts=10,
        parameters=[rfsoc.Parameter.BIAS],
        starts=[-0.5],
        stops=[0.4],
        indexes=[0],
    )
    assert targ.expts == rfsoc_sweeper.expts
    assert targ.parameters == rfsoc_sweeper.parameters
    assert targ.starts == rfsoc_sweeper.starts
    assert targ.stops == np.round(rfsoc_sweeper.stops, 2)
    assert targ.indexes == rfsoc_sweeper.indexes
    sweeper = Sweeper(
        parameter=Parameter.bias,
        values=np.arange(-0.5, +0.5, 0.1),
        qubits=[qubit],
        type=SweeperType.OFFSET,
    )
    rfsoc_sweeper = convert(sweeper, seq, platform.qubits)
    targ = rfsoc.Sweeper(
        expts=10,
        parameters=[rfsoc.Parameter.BIAS],
        starts=[-0.45],
        stops=[0.45],
        indexes=[0],
    )
    assert targ.expts == rfsoc_sweeper.expts
    assert targ.parameters == rfsoc_sweeper.parameters
    assert targ.starts == rfsoc_sweeper.starts
    assert targ.stops == np.round(rfsoc_sweeper.stops, 2)
    assert targ.indexes == rfsoc_sweeper.indexes

    qubit.flux.offset = 0.5
    sweeper = Sweeper(
        parameter=Parameter.bias,
        values=np.arange(0, +1, 0.1),
        qubits=[qubit],
        type=SweeperType.OFFSET,
    )
    with pytest.raises(ValueError):
        rfsoc_sweeper = convert(sweeper, seq, platform.qubits)

    sweeper = Sweeper(
        parameter=Parameter.frequency, values=np.arange(0, 100, 1), pulses=[pulse0]
    )
    rfsoc_sweeper = convert(sweeper, seq, platform.qubits)
    targ = rfsoc.Sweeper(
        expts=100,
        parameters=[rfsoc.Parameter.FREQUENCY],
        starts=[0],
        stops=[99],
        indexes=[0],
    )
    assert rfsoc_sweeper == targ

    sweeper = Sweeper(
        parameter=Parameter.duration, values=np.arange(40, 100, 1), pulses=[pulse0]
    )
    rfsoc_sweeper = convert(sweeper, seq, platform.qubits)
    targ = rfsoc.Sweeper(
        expts=60,
        parameters=[rfsoc.Parameter.DURATION, rfsoc.Parameter.DELAY],
        starts=[40, 40],
        stops=[99, 99],
        indexes=[0, 1],
    )
    assert (rfsoc_sweeper.starts == targ.starts).all()
    assert (rfsoc_sweeper.stops == targ.stops).all()
    assert rfsoc_sweeper.expts == targ.expts
    assert rfsoc_sweeper.parameters == targ.parameters
    assert rfsoc_sweeper.indexes == targ.indexes

    sweeper = Sweeper(
        parameter=Parameter.start, values=np.arange(0, 10, 1), pulses=[pulse0]
    )
    rfsoc_sweeper = convert(sweeper, seq, platform.qubits)
    targ = rfsoc.Sweeper(
        expts=10,
        parameters=[rfsoc.Parameter.DELAY],
        starts=[0],
        stops=[9],
        indexes=[0],
    )
    assert rfsoc_sweeper == targ


def test_rfsoc_init(dummy_qrc):
    """Tests instrument can initilize and its attribute are assigned."""
    platform = create_platform("rfsoc")
    instrument = platform.instruments["tii_rfsoc4x2"]

    assert instrument.host == "0.0.0.0"
    assert instrument.port == 0
    assert isinstance(instrument.cfg, rfsoc.Config)


def test_play(mocker, dummy_qrc):
    platform = create_platform("rfsoc")
    instrument = platform.instruments["tii_rfsoc4x2"]

    seq = PulseSequence()
    pulse0 = Pulse(0, 40, 0.9, 50e6, 0, Gaussian(2), 0, PulseType.DRIVE, 0)
    pulse1 = Pulse(40, 40, 0.9, 50e6, 0, Rectangular(), 0, PulseType.READOUT, 0)
    seq.append(pulse0)
    seq.append(pulse1)

    nshots = 100
    server_results = ([[np.random.rand(nshots)]], [[np.random.rand(nshots)]])
    mocker.patch("qibosoq.client.connect", return_value=server_results)
    parameters = ExecutionParameters(
        nshots=nshots,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )
    results = instrument.play(platform.qubits, platform.couplers, seq, parameters)
    assert pulse1.serial in results.keys()

    parameters = ExecutionParameters(
        nshots=nshots,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )
    results = instrument.play(platform.qubits, platform.couplers, seq, parameters)
    assert pulse1.serial in results.keys()

    parameters = ExecutionParameters(
        nshots=nshots,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    results = instrument.play(platform.qubits, platform.couplers, seq, parameters)
    assert pulse1.serial in results.keys()


def test_sweep(mocker, dummy_qrc):
    platform = create_platform("rfsoc")
    instrument = platform.instruments["tii_rfsoc4x2"]
    qubit = platform.qubits[0]
    qubit.flux.offset = 0.05
    qubit.flux.ports = [("name", 4)]

    seq = PulseSequence()
    pulse0 = Pulse(0, 40, 0.9, 50e6, 0, Gaussian(2), 0, PulseType.DRIVE, 0)
    pulse1 = Pulse(40, 40, 0.9, 50e6, 0, Rectangular(), 0, PulseType.READOUT, 0)
    seq.append(pulse0)
    seq.append(pulse1)
    sweeper0 = Sweeper(
        parameter=Parameter.frequency, values=np.arange(0, 100, 1), pulses=[pulse0]
    )
    sweeper1 = Sweeper(
        parameter=Parameter.bias, values=np.arange(0, 0.1, 0.01), qubits=[qubit]
    )

    nshots = 100
    server_results = ([[np.random.rand(nshots)]], [[np.random.rand(nshots)]])
    mocker.patch("qibosoq.client.connect", return_value=server_results)
    parameters = ExecutionParameters(
        nshots=nshots,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )
    results = instrument.sweep(
        platform.qubits, platform.couplers, seq, parameters, sweeper0, sweeper1
    )
    assert pulse1.serial in results.keys()

    parameters = ExecutionParameters(
        nshots=nshots,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )
    results = instrument.sweep(
        platform.qubits, platform.couplers, seq, parameters, sweeper0, sweeper1
    )
    assert pulse1.serial in results.keys()

    parameters = ExecutionParameters(
        nshots=nshots,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    results = instrument.sweep(
        platform.qubits, platform.couplers, seq, parameters, sweeper0, sweeper1
    )
    assert pulse1.serial in results.keys()


def test_validate_input_command(dummy_qrc):
    platform = create_platform("rfsoc")
    instrument = platform.instruments["tii_rfsoc4x2"]

    seq = PulseSequence()
    pulse0 = Pulse(0, 40, 0.9, 50e6, 0, Gaussian(2), 0, PulseType.DRIVE, 0)
    pulse1 = Pulse(40, 40, 0.9, 50e6, 0, Rectangular(), 0, PulseType.READOUT, 0)
    seq.append(pulse0)
    seq.append(pulse1)

    parameters = ExecutionParameters(acquisition_type=AcquisitionType.RAW)
    with pytest.raises(NotImplementedError):
        results = instrument.play(platform.qubits, platform.couplers, seq, parameters)

    parameters = ExecutionParameters(fast_reset=True)
    with pytest.raises(NotImplementedError):
        results = instrument.play(platform.qubits, platform.couplers, seq, parameters)


def test_update_cfg(mocker, dummy_qrc):
    platform = create_platform("rfsoc")
    instrument = platform.instruments["tii_rfsoc4x2"]

    seq = PulseSequence()
    pulse0 = Pulse(0, 40, 0.9, 50e6, 0, Gaussian(2), 0, PulseType.DRIVE, 0)
    pulse1 = Pulse(40, 40, 0.9, 50e6, 0, Rectangular(), 0, PulseType.READOUT, 0)
    seq.append(pulse0)
    seq.append(pulse1)

    nshots = 333
    relax_time = 1e6
    server_results = ([[np.random.rand(nshots)]], [[np.random.rand(nshots)]])
    mocker.patch("qibosoq.client.connect", return_value=server_results)
    parameters = ExecutionParameters(
        nshots=nshots,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
        relaxation_time=relax_time,
    )
    results = instrument.play(platform.qubits, platform.couplers, seq, parameters)
    assert instrument.cfg.reps == nshots
    relax_time = relax_time * 1e-3
    assert instrument.cfg.repetition_duration == relax_time


def test_classify_shots(dummy_qrc):
    """Creates fake IQ values and check classification works as expected."""
    qubit0 = Qubit(name="q0", threshold=1, iq_angle=np.pi / 2)
    qubit1 = Qubit(
        name="q1",
    )
    i_val = [0] * 7
    q_val = [-5, -1.5, -0.5, 0, 0.5, 1.5, 5]

    platform = create_platform("rfsoc")
    instrument = platform.instruments["tii_rfsoc4x2"]

    shots = instrument.classify_shots(i_val, q_val, qubit0)
    target_shots = np.array([1, 1, 0, 0, 0, 0, 0])

    assert (target_shots == shots).all()
    with pytest.raises(ValueError):
        instrument.classify_shots(i_val, q_val, qubit1)


def test_merge_sweep_results(dummy_qrc):
    """Creates fake dictionary of results and check merging works as
    expected."""
    dict_a = {"serial1": AveragedIntegratedResults(np.array([0 + 1j * 1]))}
    dict_b = {
        "serial1": AveragedIntegratedResults(np.array([4 + 1j * 4])),
        "serial2": AveragedIntegratedResults(np.array([5 + 1j * 5])),
    }
    dict_c = {}
    targ_dict = {
        "serial1": AveragedIntegratedResults(np.array([0 + 1j * 1, 4 + 1j * 4])),
        "serial2": AveragedIntegratedResults(np.array([5 + 1j * 5])),
    }

    platform = create_platform("rfsoc")
    instrument = platform.instruments["tii_rfsoc4x2"]

    out_dict1 = instrument.merge_sweep_results(dict_a, dict_b)
    out_dict2 = instrument.merge_sweep_results(dict_c, dict_a)

    assert targ_dict.keys() == out_dict1.keys()
    assert (
        out_dict1["serial1"].serialize["MSR[V]"]
        == targ_dict["serial1"].serialize["MSR[V]"]
    ).all()
    assert (
        out_dict1["serial1"].serialize["MSR[V]"]
        == targ_dict["serial1"].serialize["MSR[V]"]
    ).all()

    assert dict_a.keys() == out_dict2.keys()
    assert (
        out_dict2["serial1"].serialize["MSR[V]"]
        == dict_a["serial1"].serialize["MSR[V]"]
    ).all()
    assert (
        out_dict2["serial1"].serialize["MSR[V]"]
        == dict_a["serial1"].serialize["MSR[V]"]
    ).all()


def test_get_if_python_sweep(dummy_qrc):
    """Creates pulse sequences and check if they can be swept by the firmware.

    Qibosoq does not support sweep on readout frequency, more than one
    sweep at the same time, sweep on channels where multiple pulses are
    sent. If Qibosoq does not support the sweep, the driver will use a
    python loop
    """

    platform = create_platform("rfsoc")
    instrument = platform.instruments["tii_rfsoc4x2"]

    sequence_1 = PulseSequence()
    sequence_1.append(platform.create_RX_pulse(qubit=0, start=0))
    sequence_1.append(platform.create_MZ_pulse(qubit=0, start=100))

    sweep1 = Sweeper(
        parameter=Parameter.frequency,
        values=np.arange(10, 100, 10),
        pulses=[sequence_1[0]],
    )
    sweep2 = Sweeper(
        parameter=Parameter.frequency,
        values=np.arange(10, 100, 10),
        pulses=[sequence_1[1]],
    )
    sweep3 = Sweeper(
        parameter=Parameter.amplitude,
        values=np.arange(0.01, 0.5, 0.1),
        pulses=[sequence_1[1]],
    )
    sweep1 = convert(sweep1, sequence_1, platform.qubits)
    sweep2 = convert(sweep2, sequence_1, platform.qubits)
    sweep3 = convert(sweep3, sequence_1, platform.qubits)

    assert instrument.get_if_python_sweep(sequence_1, sweep2)
    assert not instrument.get_if_python_sweep(sequence_1, sweep1)
    assert not instrument.get_if_python_sweep(sequence_1, sweep3)

    sequence_2 = PulseSequence()
    sequence_2.append(platform.create_RX_pulse(qubit=0, start=0))

    sweep1 = Sweeper(
        parameter=Parameter.frequency,
        values=np.arange(10, 100, 10),
        pulses=[sequence_2[0]],
    )
    sweep2 = Sweeper(
        parameter=Parameter.amplitude,
        values=np.arange(0.01, 0.5, 0.1),
        pulses=[sequence_2[0]],
    )
    sweep1 = convert(sweep1, sequence_2, platform.qubits)
    sweep2 = convert(sweep2, sequence_2, platform.qubits)

    assert not instrument.get_if_python_sweep(sequence_2, sweep1)
    assert not instrument.get_if_python_sweep(sequence_2, sweep1, sweep2)

    # TODO repetition
    platform = create_platform("rfsoc")
    instrument = platform.instruments["tii_rfsoc4x2"]

    sequence_1 = PulseSequence()
    sequence_1.append(platform.create_RX_pulse(qubit=0, start=0))
    sweep1 = Sweeper(
        parameter=Parameter.frequency,
        values=np.arange(10, 100, 10),
        pulses=[sequence_1[0]],
    )
    sweep2 = Sweeper(
        parameter=Parameter.relative_phase,
        values=np.arange(0, 1, 0.01),
        pulses=[sequence_1[0]],
    )
    sweep3 = Sweeper(
        parameter=Parameter.bias,
        values=np.arange(-0.1, 0.1, 0.001),
        qubits=[platform.qubits[0]],
    )
    sweep1 = convert(sweep1, sequence_1, platform.qubits)
    sweep2 = convert(sweep2, sequence_1, platform.qubits)
    sweep3 = convert(sweep3, sequence_1, platform.qubits)
    assert not instrument.get_if_python_sweep(sequence_1, sweep1, sweep2, sweep3)

    platform.qubits[0].flux.offset = 0.5
    sweep1 = Sweeper(parameter=Parameter.bias, values=np.arange(-1, 1, 0.1), qubits=[0])
    with pytest.raises(ValueError):
        sweep1 = convert(sweep1, sequence_1, platform.qubits)


def test_convert_av_sweep_results(dummy_qrc):
    """Qibosoq sends results using nested lists, check if the conversion to
    dictionary of AveragedResults, for averaged sweep, works as expected."""

    platform = create_platform("rfsoc")
    instrument = platform.instruments["tii_rfsoc4x2"]

    sequence = PulseSequence()
    sequence.append(platform.create_RX_pulse(qubit=0, start=0))
    sequence.append(platform.create_MZ_pulse(qubit=0, start=100))
    sequence.append(platform.create_MZ_pulse(qubit=0, start=200))
    sweep1 = Sweeper(
        parameter=Parameter.frequency,
        values=np.arange(10, 35, 10),
        pulses=[sequence[0]],
    )
    sweep1 = convert(sweep1, sequence, platform.qubits)
    serial1 = sequence[1].serial
    serial2 = sequence[2].serial

    avgi = [[[1, 2, 3], [4, 1, 2]]]
    avgq = [[[7, 8, 9], [-1, -2, -3]]]

    execution_parameters = ExecutionParameters(
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    out_dict = instrument.convert_sweep_results(
        sequence.ro_pulses, platform.qubits, avgi, avgq, execution_parameters
    )
    targ_dict = {
        serial1: AveragedIntegratedResults(
            np.array([1, 2, 3]) + 1j * np.array([7, 8, 9])
        ),
        serial2: AveragedIntegratedResults(
            np.array([4, 1, 2]) + 1j * np.array([-1, -2, -3])
        ),
    }

    assert (
        out_dict[serial1].serialize["i[V]"] == targ_dict[serial1].serialize["i[V]"]
    ).all()
    assert (
        out_dict[serial1].serialize["q[V]"] == targ_dict[serial1].serialize["q[V]"]
    ).all()
    assert (
        out_dict[serial2].serialize["i[V]"] == targ_dict[serial2].serialize["i[V]"]
    ).all()
    assert (
        out_dict[serial2].serialize["q[V]"] == targ_dict[serial2].serialize["q[V]"]
    ).all()


def test_convert_nav_sweep_results(dummy_qrc):
    """Qibosoq sends results using nested lists, check if the conversion to
    dictionary of ExecutionResults, for not averaged sweep, works as
    expected."""
    platform = create_platform("rfsoc")
    instrument = platform.instruments["tii_rfsoc4x2"]

    sequence = PulseSequence()
    sequence.append(platform.create_RX_pulse(qubit=0, start=0))
    sequence.append(platform.create_MZ_pulse(qubit=0, start=100))
    sequence.append(platform.create_MZ_pulse(qubit=0, start=200))
    sweep1 = Sweeper(
        parameter=Parameter.frequency,
        values=np.arange(10, 35, 10),
        pulses=[sequence[0]],
    )
    sweep1 = convert(sweep1, sequence, platform.qubits)
    serial1 = sequence[1].serial
    serial2 = sequence[2].serial

    avgi = [[[[1, 1], [2, 2], [3, 3]], [[4, 4], [1, 1], [2, 2]]]]
    avgq = [[[[7, 7], [8, 8], [9, 9]], [[-1, -1], [-2, -2], [-3, -3]]]]

    execution_parameters = ExecutionParameters(
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    out_dict = instrument.convert_sweep_results(
        sequence.ro_pulses, platform.qubits, avgi, avgq, execution_parameters
    )
    targ_dict = {
        serial1: AveragedIntegratedResults(
            np.array([1, 1, 2, 2, 3, 3]) + 1j * np.array([7, 7, 8, 8, 9, 9])
        ),
        serial2: AveragedIntegratedResults(
            np.array([4, 4, 1, 1, 2, 2]) + 1j * np.array([-1, -1, -2, -2, -3, -3])
        ),
    }

    assert (
        out_dict[serial1].serialize["i[V]"] == targ_dict[serial1].serialize["i[V]"]
    ).all()
    assert (
        out_dict[serial1].serialize["q[V]"] == targ_dict[serial1].serialize["q[V]"]
    ).all()
    assert (
        out_dict[serial2].serialize["i[V]"] == targ_dict[serial2].serialize["i[V]"]
    ).all()
    assert (
        out_dict[serial2].serialize["q[V]"] == targ_dict[serial2].serialize["q[V]"]
    ).all()


@pytest.fixture(scope="module")
def instrument(connected_platform):
    return get_instrument(connected_platform, RFSoC)


@pytest.mark.qpu
def test_call_executepulsesequence(connected_platform, instrument):
    """Executes a PulseSequence and check if result shape is as expected.

    Both for averaged results and not averaged results.
    """
    platform = connected_platform
    instrument = platform.instruments["tii_rfsoc4x2"]

    sequence = PulseSequence()
    sequence.append(platform.create_RX_pulse(qubit=0, start=0))
    sequence.append(platform.create_MZ_pulse(qubit=0, start=100))

    instrument.cfg.average = False
    i_vals_nav, q_vals_nav = instrument._execute_pulse_sequence(
        sequence, platform.qubits, rfsoc.OperationCode.EXECUTE_PULSE_SEQUENCE
    )
    instrument.cfg.average = True
    i_vals_av, q_vals_av = instrument._execute_pulse_sequence(
        sequence, platform.qubits, rfsoc.OperationCode.EXECUTE_PULSE_SEQUENCE
    )

    assert np.shape(i_vals_nav) == (1, 1, 1000)
    assert np.shape(q_vals_nav) == (1, 1, 1000)
    assert np.shape(i_vals_av) == (1, 1)
    assert np.shape(q_vals_av) == (1, 1)


@pytest.mark.qpu
def test_call_execute_sweeps(connected_platform, instrument):
    """Execute a firmware sweep and check if result shape is as expected.

    Both for averaged results and not averaged results.
    """
    platform = connected_platform
    instrument = platform.instruments["tii_rfsoc4x2"]

    sequence = PulseSequence()
    sequence.append(platform.create_RX_pulse(qubit=0, start=0))
    sequence.append(platform.create_MZ_pulse(qubit=0, start=100))
    sweep = Sweeper(
        parameter=Parameter.frequency,
        values=np.arange(10, 35, 10),
        pulses=[sequence[0]],
    )
    expts = len(sweep.values)

    sweep = [convert(sweep, sequence, platform.qubits)]
    instrument.cfg.average = False
    i_vals_nav, q_vals_nav = instrument._execute_sweeps(
        sequence, platform.qubits, sweep
    )
    instrument.cfg.average = True
    i_vals_av, q_vals_av = instrument._execute_sweeps(sequence, platform.qubits, sweep)

    assert np.shape(i_vals_nav) == (1, 1, expts, 1000)
    assert np.shape(q_vals_nav) == (1, 1, expts, 1000)
    assert np.shape(i_vals_av) == (1, 1, expts)
    assert np.shape(q_vals_av) == (1, 1, expts)


@pytest.mark.qpu
def test_play_qpu(connected_platform, instrument):
    """Sends a PulseSequence using `play` and check results are what
    expected."""
    platform = connected_platform
    instrument = platform.instruments["tii_rfsoc4x2"]

    sequence = PulseSequence()
    sequence.append(platform.create_RX_pulse(qubit=0, start=0))
    sequence.append(platform.create_MZ_pulse(qubit=0, start=100))

    out_dict = instrument.play(
        platform.qubits,
        sequence,
        ExecutionParameters(acquisition_type=AcquisitionType.INTEGRATION),
    )

    assert sequence[1].serial in out_dict
    assert isinstance(out_dict[sequence[1].serial], IntegratedResults)
    assert np.shape(out_dict[sequence[1].serial].voltage_i) == (1000,)


@pytest.mark.qpu
def test_sweep_qpu(connected_platform, instrument):
    """Sends a PulseSequence using `sweep` and check results are what
    expected."""
    platform = connected_platform
    instrument = platform.instruments["tii_rfsoc4x2"]

    sequence = PulseSequence()
    sequence.append(platform.create_RX_pulse(qubit=0, start=0))
    sequence.append(platform.create_MZ_pulse(qubit=0, start=100))
    sweep = Sweeper(
        parameter=Parameter.frequency,
        values=np.arange(10, 35, 10),
        pulses=[sequence[0]],
    )

    out_dict1 = instrument.sweep(
        platform.qubits,
        platform.couplers,
        sequence,
        ExecutionParameters(
            relaxation_time=100_000, averaging_mode=AveragingMode.CYCLIC
        ),
        sweep,
    )
    out_dict2 = instrument.sweep(
        platform.qubits,
        platform.couplers,
        sequence,
        ExecutionParameters(
            relaxation_time=100_000,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.SINGLESHOT,
        ),
        sweep,
    )

    assert sequence[1].serial in out_dict1
    assert sequence[1].serial in out_dict2
    assert isinstance(out_dict1[sequence[1].serial], AveragedSampleResults)
    assert isinstance(out_dict2[sequence[1].serial], IntegratedResults)
    assert np.shape(out_dict2[sequence[1].serial].voltage_i) == (
        1000,
        len(sweep.values),
    )
    assert np.shape(out_dict1[sequence[1].serial].statistical_frequency) == (
        len(sweep.values),
    )


@pytest.mark.qpu
def test_python_reqursive_sweep(connected_platform, instrument):
    """Sends a PulseSequence directly to `python_reqursive_sweep` and check
    results are what expected."""
    platform = connected_platform
    instrument = platform.instruments["tii_rfsoc4x2"]

    sequence = PulseSequence()
    sequence.append(platform.create_RX_pulse(qubit=0, start=0))
    sequence.append(platform.create_MZ_pulse(qubit=0, start=100))
    sweep1 = Sweeper(
        parameter=Parameter.amplitude,
        values=np.arange(0.01, 0.03, 10),
        pulses=[sequence[0]],
    )
    sweep2 = Sweeper(
        parameter=Parameter.frequency,
        values=np.arange(10, 35, 10),
        pulses=[sequence[0]],
    )

    out_dict = instrument.sweep(
        platform.qubits,
        platform.couplers,
        sequence,
        ExecutionParameters(
            relaxation_time=100_000, averaging_mode=AveragingMode.CYCLIC
        ),
        sweep1,
        sweep2,
    )

    assert sequence[1].serial in out_dict
