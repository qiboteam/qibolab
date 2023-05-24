import numpy as np
import pytest

from qibolab import create_platform
from qibolab.instruments.rfsoc import QickProgramConfig
from qibolab.paths import qibolab_folder
from qibolab.platform import Qubit
from qibolab.pulses import PulseSequence
from qibolab.result import AveragedResults, ExecutionResults
from qibolab.sweeper import Parameter, Sweeper


def test_tii_rfsoc4x2_init():
    """Tests instrument can initilize and its attribute are assigned"""
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    assert instrument.host == "0.0.0.0"
    assert instrument.port == 0
    assert isinstance(instrument.cfg, QickProgramConfig)


def test_tii_rfsoc4x2_setup():
    """Modify the QickProgramConfig object using `setup` and check that it changes accordingly"""
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    target_cfg = QickProgramConfig(
        sampling_rate=5_000_000_000, repetition_duration=1_000, adc_trig_offset=150, max_gain=30_000
    )

    instrument.setup(sampling_rate=5_000_000_000, relaxation_time=1_000, adc_trig_offset=150, max_gain=30_000)

    assert instrument.cfg == target_cfg


def test_classify_shots():
    """Creates fake IQ values and check classification works as expected"""
    qubit0 = Qubit(name="q0", threshold=1, iq_angle=np.pi / 2)
    qubit1 = Qubit(
        name="q1",
    )
    i_val = [0] * 7
    q_val = [-5, -1.5, -0.5, 0, 0.5, 1.5, 5]

    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    shots = instrument.classify_shots(i_val, q_val, qubit0)
    target_shots = np.array([1, 1, 0, 0, 0, 0, 0])

    assert (target_shots == shots).all()
    assert instrument.classify_shots(i_val, q_val, qubit1) is None


def test_merge_sweep_results():
    """Creates fake dictionary of results and check merging works as expected"""
    dict_a = {"serial1": AveragedResults.from_components(np.array([0]), np.array([1]))}
    dict_b = {
        "serial1": AveragedResults.from_components(np.array([4]), np.array([4])),
        "serial2": AveragedResults.from_components(np.array([5]), np.array([5])),
    }
    dict_c = {}
    targ_dict = {
        "serial1": AveragedResults.from_components(np.array([0, 4]), np.array([1, 4])),
        "serial2": AveragedResults.from_components(np.array([5]), np.array([5])),
    }

    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]
    out_dict1 = instrument.merge_sweep_results(dict_a, dict_b)
    out_dict2 = instrument.merge_sweep_results(dict_c, dict_a)

    assert targ_dict.keys() == out_dict1.keys()
    assert (out_dict1["serial1"].i == targ_dict["serial1"].i).all()
    assert (out_dict1["serial1"].q == targ_dict["serial1"].q).all()

    assert dict_a.keys() == out_dict2.keys()
    assert (out_dict2["serial1"].i == dict_a["serial1"].i).all()
    assert (out_dict2["serial1"].q == dict_a["serial1"].q).all()


def test_get_if_python_sweep():
    """Creates pulse sequences and check if they can be swept by the firmware.

    Qibosoq does not support sweep on readout frequency, more than one sweep
    at the same time, sweep on channels where multiple pulses are sent.
    If Qibosoq does not support the sweep, the driver will use a python loop
    """
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence_1 = PulseSequence()
    sequence_1.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence_1.add(platform.create_MZ_pulse(qubit=0, start=100))
    sweep1 = Sweeper(parameter=Parameter.frequency, values=np.arange(10, 100, 10), pulses=[sequence_1[0]])
    sweep2 = Sweeper(parameter=Parameter.frequency, values=np.arange(10, 100, 10), pulses=[sequence_1[1]])
    sweep3 = Sweeper(parameter=Parameter.amplitude, values=np.arange(0.01, 0.5, 0.1), pulses=[sequence_1[1]])
    sequence_2 = PulseSequence()
    sequence_2.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence_2.add(platform.create_RX_pulse(qubit=0, start=100))

    assert instrument.get_if_python_sweep(sequence_1, platform.qubits, sweep2)
    assert instrument.get_if_python_sweep(sequence_2, platform.qubits, sweep1)
    assert instrument.get_if_python_sweep(sequence_2, platform.qubits, sweep1, sweep1)
    assert not instrument.get_if_python_sweep(sequence_1, platform.qubits, sweep1)
    assert not instrument.get_if_python_sweep(sequence_1, platform.qubits, sweep3)


def test_convert_av_sweep_results():
    """Qibosoq sends results using nested lists, check if the conversion
    to dictionary of AveragedResults, for averaged sweep, works as expected
    """
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=100))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=200))
    sweep1 = Sweeper(parameter=Parameter.frequency, values=np.arange(10, 35, 10), pulses=[sequence[0]])
    serial1 = sequence[1].serial
    serial2 = sequence[2].serial

    avgi = [[[1, 2, 3], [0, 1, 2]]]
    avgq = [[[7, 8, 9], [-1, -2, -3]]]

    ro_serials = [ro.serial for ro in sequence.ro_pulses]
    out_dict = instrument.convert_sweep_results(sweep1, ro_serials, sequence, platform.qubits, avgi, avgq, True)
    targ_dict = {
        serial1: AveragedResults.from_components(np.array([1, 2, 3]), np.array([7, 8, 9])),
        serial2: AveragedResults.from_components(np.array([0, 1, 2]), np.array([-1, -2, -3])),
    }

    assert (out_dict[serial1].i == targ_dict[serial1].i).all()
    assert (out_dict[serial1].q == targ_dict[serial1].q).all()
    assert (out_dict[serial2].i == targ_dict[serial2].i).all()
    assert (out_dict[serial2].q == targ_dict[serial2].q).all()


def test_convert_nav_sweep_results():
    """Qibosoq sends results using nested lists, check if the conversion
    to dictionary of ExecutionResults, for not averaged sweep, works as expected
    """
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=100))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=200))
    sweep1 = Sweeper(parameter=Parameter.frequency, values=np.arange(10, 35, 10), pulses=[sequence[0]])
    serial1 = sequence[1].serial
    serial2 = sequence[2].serial

    avgi = [[[[1, 1], [2, 2], [3, 3]], [[0, 0], [1, 1], [2, 2]]]]
    avgq = [[[[7, 7], [8, 8], [9, 9]], [[-1, -1], [-2, -2], [-3, -3]]]]

    ro_serials = [ro.serial for ro in sequence.ro_pulses]
    out_dict = instrument.convert_sweep_results(sweep1, ro_serials, sequence, platform.qubits, avgi, avgq, False)
    targ_dict = {
        serial1: ExecutionResults.from_components(np.array([1, 1, 2, 2, 3, 3]), np.array([7, 7, 8, 8, 9, 9])),
        serial2: ExecutionResults.from_components(np.array([0, 0, 1, 1, 2, 2]), np.array([-1, -1, -2, -2, -3, -3])),
    }

    assert (out_dict[serial1].i == targ_dict[serial1].i).all()
    assert (out_dict[serial1].q == targ_dict[serial1].q).all()
    assert (out_dict[serial2].i == targ_dict[serial2].i).all()
    assert (out_dict[serial2].q == targ_dict[serial2].q).all()


@pytest.mark.qpu
def test_call_executepulsesequence():
    """Executes a PulseSequence and check if result shape is as expected.
    Both for averaged results and not averaged results.
    """
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=100))

    i_vals_nav, q_vals_nav = instrument._execute_pulse_sequence(instrument.cfg, sequence, platform.qubits, 1, False)
    i_vals_av, q_vals_av = instrument._execute_pulse_sequence(instrument.cfg, sequence, platform.qubits, 1, True)

    assert np.shape(i_vals_nav) == (1, 1, 1000)
    assert np.shape(q_vals_nav) == (1, 1, 1000)
    assert np.shape(i_vals_av) == (1, 1)
    assert np.shape(q_vals_av) == (1, 1)


@pytest.mark.qpu
def test_call_executesinglesweep():
    """Executes a firmware sweep and check if result shape is as expected.
    Both for averaged results and not averaged results.
    """
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=100))
    sweep = Sweeper(parameter=Parameter.frequency, values=np.arange(10, 35, 10), pulses=[sequence[0]])

    i_vals_nav, q_vals_nav = instrument._execute_single_sweep(
        instrument.cfg, sequence, platform.qubits, sweep, 1, False
    )
    i_vals_av, q_vals_av = instrument._execute_single_sweep(instrument.cfg, sequence, platform.qubits, sweep, 1, True)

    assert np.shape(i_vals_nav) == (1, 1, len(sweep.values), 1000)
    assert np.shape(q_vals_nav) == (1, 1, len(sweep.values), 1000)
    assert np.shape(i_vals_av) == (1, 1, len(sweep.values))
    assert np.shape(q_vals_av) == (1, 1, len(sweep.values))


@pytest.mark.qpu
def test_play():
    """Sends a PulseSequence using `play` and check results are what expected"""
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=100))

    out_dict = instrument.play(platform.qubits, sequence)

    assert sequence[1].serial in out_dict
    assert isinstance(out_dict[sequence[1].serial], ExecutionResults)
    assert np.shape(out_dict[sequence[1].serial].i) == (1000,)


@pytest.mark.qpu
def test_sweep():
    """Sends a PulseSequence using `sweep` and check results are what expected"""
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=100))
    sweep = Sweeper(parameter=Parameter.frequency, values=np.arange(10, 35, 10), pulses=[sequence[0]])

    out_dict1 = instrument.sweep(platform.qubits, sequence, sweep, average=True, relaxation_time=100_000)
    out_dict2 = instrument.sweep(platform.qubits, sequence, sweep, average=False, relaxation_time=100_000)

    assert sequence[1].serial in out_dict1
    assert sequence[1].serial in out_dict2
    assert isinstance(out_dict1[sequence[1].serial], AveragedResults)
    assert isinstance(out_dict2[sequence[1].serial], ExecutionResults)
    assert np.shape(out_dict1[sequence[1].serial].i) == (len(sweep.values),)
    assert np.shape(out_dict2[sequence[1].serial].i) == (len(sweep.values) * 1000,)


@pytest.mark.qpu
def test_python_reqursive_sweep():
    """Sends a PulseSequence directly to `python_reqursive_sweep` and check results are what expected"""
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=100))
    sweep1 = Sweeper(parameter=Parameter.amplitude, values=np.arange(0.01, 0.03, 10), pulses=[sequence[0]])
    sweep2 = Sweeper(parameter=Parameter.frequency, values=np.arange(10, 35, 10), pulses=[sequence[0]])

    out_dict = instrument.sweep(platform.qubits, sequence, sweep1, sweep2, average=True, relaxation_time=100_000)

    assert sequence[1].serial in out_dict
