from unittest.mock import patch

import numpy as np
import pytest
from qm import qua

from qibolab import AcquisitionType, ExecutionParameters, create_platform
from qibolab.instruments.qm import OPXplus, QMController
from qibolab.instruments.qm.acquisition import Acquisition, declare_acquisitions
from qibolab.instruments.qm.controller import controllers_config
from qibolab.instruments.qm.sequence import BakedPulse, QMPulse, Sequence
from qibolab.pulses import FluxPulse, Pulse, PulseSequence, ReadoutPulse, Rectangular
from qibolab.qubits import Qubit
from qibolab.sweeper import Parameter, Sweeper

from .conftest import set_platform_profile


def test_qmpulse():
    pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    qmpulse = QMPulse(pulse)
    assert qmpulse.operation == "drive(40, 0.05, Rectangular())"
    assert qmpulse.relative_phase == 0


@pytest.mark.parametrize("acquisition_type", AcquisitionType)
def test_qmpulse_declare_output(acquisition_type):
    options = ExecutionParameters(acquisition_type=acquisition_type)
    pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    qmpulse = QMPulse(pulse)
    qubits = {0: Qubit(0, threshold=0.1, iq_angle=0.2)}
    if acquisition_type is AcquisitionType.SPECTROSCOPY:
        with pytest.raises(KeyError):
            with qua.program() as _:
                declare_acquisitions([qmpulse], qubits, options)
    else:
        with qua.program() as _:
            declare_acquisitions([qmpulse], qubits, options)
        acquisition = qmpulse.acquisition
        assert isinstance(acquisition, Acquisition)
        if acquisition_type is AcquisitionType.DISCRIMINATION:
            assert acquisition.threshold == 0.1
            assert acquisition.cos == np.cos(0.2)
            assert acquisition.sin == np.sin(0.2)
            assert isinstance(acquisition.shot, qua._dsl._Variable)
            assert isinstance(acquisition.shots, qua._dsl._ResultSource)
        elif acquisition_type is AcquisitionType.INTEGRATION:
            assert isinstance(acquisition.i, qua._dsl._Variable)
            assert isinstance(acquisition.q, qua._dsl._Variable)
            assert isinstance(acquisition.istream, qua._dsl._ResultSource)
            assert isinstance(acquisition.qstream, qua._dsl._ResultSource)
        elif acquisition_type is AcquisitionType.RAW:
            assert isinstance(acquisition.adc_stream, qua._dsl._ResultSource)


def test_qmsequence():
    qd_pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    ro_pulse = ReadoutPulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch1", qubit=0)
    qmsequence = Sequence()
    with pytest.raises(AttributeError):
        qmsequence.add("test")
    qmsequence.add(QMPulse(qd_pulse))
    qmsequence.add(QMPulse(ro_pulse))
    assert len(qmsequence.pulse_to_qmpulse) == 2
    assert len(qmsequence.qmpulses) == 2


def test_qmpulse_previous_and_next():
    nqubits = 5
    qmsequence = Sequence()
    qd_qmpulses = []
    ro_qmpulses = []
    for qubit in range(nqubits):
        qd_pulse = QMPulse(
            Pulse(
                0, 40, 0.05, int(3e9), 0.0, Rectangular(), f"drive{qubit}", qubit=qubit
            )
        )
        qd_qmpulses.append(qd_pulse)
        qmsequence.add(qd_pulse)
    for qubit in range(nqubits):
        ro_pulse = QMPulse(
            ReadoutPulse(
                40,
                100,
                0.05,
                int(3e9),
                0.0,
                Rectangular(),
                f"readout{qubit}",
                qubit=qubit,
            )
        )
        ro_qmpulses.append(ro_pulse)
        qmsequence.add(ro_pulse)

    for qd_qmpulse, ro_qmpulse in zip(qd_qmpulses, ro_qmpulses):
        assert len(qd_qmpulse.next_) == 1
        assert len(ro_qmpulse.next_) == 0


def test_qmpulse_previous_and_next_flux():
    y90_pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), f"drive1", qubit=1)
    x_pulse_start = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), f"drive2", qubit=2)
    flux_pulse = FluxPulse(
        start=y90_pulse.finish,
        duration=30,
        amplitude=0.055,
        shape=Rectangular(),
        channel="flux2",
        qubit=2,
    )
    theta_pulse = Pulse(70, 40, 0.05, int(3e9), 0.0, Rectangular(), f"drive1", qubit=1)
    x_pulse_end = Pulse(70, 40, 0.05, int(3e9), 0.0, Rectangular(), f"drive2", qubit=2)

    measure_lowfreq = ReadoutPulse(
        110, 100, 0.05, int(3e9), 0.0, Rectangular(), "readout1", qubit=1
    )
    measure_highfreq = ReadoutPulse(
        110, 100, 0.05, int(3e9), 0.0, Rectangular(), "readout2", qubit=2
    )

    drive11 = QMPulse(y90_pulse)
    drive21 = QMPulse(x_pulse_start)
    flux2 = QMPulse(flux_pulse)
    drive12 = QMPulse(theta_pulse)
    drive22 = QMPulse(x_pulse_end)
    measure1 = QMPulse(measure_lowfreq)
    measure2 = QMPulse(measure_highfreq)

    qmsequence = Sequence()
    qmsequence.add(drive11)
    qmsequence.add(drive21)
    qmsequence.add(flux2)
    qmsequence.add(drive12)
    qmsequence.add(drive22)
    qmsequence.add(measure1)
    qmsequence.add(measure2)
    assert drive11.next_ == set()
    assert drive21.next_ == {flux2}
    assert flux2.next_ == {drive12, drive22}
    assert drive12.next_ == {measure1}
    assert drive22.next_ == {measure2}


@pytest.fixture
def qmcontroller():
    name = "test"
    address = "0.0.0.0:0"
    return QMController(name, address, opxs=[OPXplus("con1")])


@pytest.mark.parametrize("offset", [0.0, 0.005])
def test_qm_register_port(qmcontroller, offset):
    port = qmcontroller.ports((("con1", 1),))
    port.offset = offset
    qmcontroller.config.register_port(port)
    controllers = qmcontroller.config.controllers
    assert controllers == {
        "con1": {
            "analog_inputs": {1: {}, 2: {}},
            "analog_outputs": {1: {"offset": offset, "filter": {}}},
            "digital_outputs": {},
        }
    }


def test_qm_register_port_filter(qmcontroller):
    port = qmcontroller.ports((("con1", 2),))
    port.offset = 0.005
    port.filter = {"feedforward": [1, -1], "feedback": [0.95]}
    qmcontroller.config.register_port(port)
    controllers = qmcontroller.config.controllers
    assert controllers == {
        "con1": {
            "analog_inputs": {1: {}, 2: {}},
            "analog_outputs": {
                2: {
                    "filter": {"feedback": [0.95], "feedforward": [1, -1]},
                    "offset": 0.005,
                }
            },
            "digital_outputs": {},
        }
    }


@pytest.fixture(params=["qm", "qm_octave"])
def qmplatform(request):
    set_platform_profile()
    return create_platform(request.param)


def test_controllers_config(qmplatform):
    config = controllers_config(list(qmplatform.qubits.values()), time_of_flight=30)
    assert len(config.controllers) == 3
    assert len(config.elements) == 10


# TODO: Test connect/disconnect


def test_qm_setup(qmplatform):
    platform = qmplatform
    controller = platform.instruments["qm"]
    assert controller.time_of_flight == 280


def test_qm_register_drive_element(qmplatform):
    platform = qmplatform
    controller = platform.instruments["qm"]
    controller.config.register_drive_element(
        platform.qubits[0], intermediate_frequency=int(1e6)
    )
    assert "drive0" in controller.config.elements
    if platform.name == "qm":
        target_element = {
            "mixInputs": {
                "I": ("con3", 2),
                "Q": ("con3", 1),
                "lo_frequency": 4700000000,
                "mixer": "mixer_drive0",
            },
            "intermediate_frequency": 1000000,
            "operations": {},
        }
        assert controller.config.elements["drive0"] == target_element
        target_mixer = [
            {
                "intermediate_frequency": 1000000,
                "lo_frequency": 4700000000,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ]
        assert controller.config.mixers["mixer_drive0"] == target_mixer
    else:
        target_element = {
            "RF_inputs": {"port": ("octave3", 1)},
            "digitalInputs": {
                "output_switch": {"buffer": 18, "delay": 57, "port": ("con3", 1)}
            },
            "intermediate_frequency": 1000000,
            "operations": {},
        }
        assert controller.config.elements["drive0"] == target_element
        assert "mixer_drive0" not in controller.config.mixers


def test_qm_register_readout_element(qmplatform):
    platform = qmplatform
    controller = platform.instruments["qm"]
    controller.config.register_readout_element(
        platform.qubits[2], int(1e6), controller.time_of_flight, controller.smearing
    )
    assert "readout2" in controller.config.elements
    if platform.name == "qm":
        target_element = {
            "mixInputs": {
                "I": ("con2", 10),
                "Q": ("con2", 9),
                "lo_frequency": 7900000000,
                "mixer": "mixer_readout2",
            },
            "intermediate_frequency": 1000000,
            "operations": {},
            "outputs": {
                "out1": ("con2", 2),
                "out2": ("con2", 1),
            },
            "time_of_flight": 280,
            "smearing": 0,
        }
        assert controller.config.elements["readout2"] == target_element
        target_mixer = [
            {
                "intermediate_frequency": 1000000,
                "lo_frequency": 7900000000,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ]
        assert controller.config.mixers["mixer_readout2"] == target_mixer
    else:
        target_element = {
            "RF_inputs": {"port": ("octave2", 5)},
            "RF_outputs": {"port": ("octave2", 1)},
            "digitalInputs": {
                "output_switch": {"buffer": 18, "delay": 57, "port": ("con2", 9)}
            },
            "intermediate_frequency": 1000000,
            "operations": {},
            "time_of_flight": 280,
            "smearing": 0,
        }
        assert controller.config.elements["readout2"] == target_element
        assert "mixer_readout2" not in controller.config.mixers


@pytest.mark.parametrize("pulse_type,qubit", [("drive", 2), ("readout", 1)])
def test_qm_register_pulse(qmplatform, pulse_type, qubit):
    platform = qmplatform
    controller = platform.instruments["qm"]
    if pulse_type == "drive":
        pulse = platform.create_RX_pulse(qubit, start=0)
        target_pulse = {
            "operation": "control",
            "length": pulse.duration,
            "digital_marker": "ON",
            "waveforms": {
                "I": hash(pulse.envelope_waveform_i()),
                "Q": hash(pulse.envelope_waveform_q()),
            },
        }

    else:
        pulse = platform.create_MZ_pulse(qubit, start=0)
        target_pulse = {
            "operation": "measurement",
            "length": pulse.duration,
            "waveforms": {"I": "constant_wf0.003575", "Q": "zero_wf"},
            "digital_marker": "ON",
            "integration_weights": {
                "cos": "cosine_weights1",
                "minus_sin": "minus_sine_weights1",
                "sin": "sine_weights1",
            },
        }

    controller.config.register_element(
        platform.qubits[qubit], pulse, controller.time_of_flight, controller.smearing
    )
    qmpulse = QMPulse(pulse)
    controller.config.register_pulse(platform.qubits[qubit], qmpulse)
    assert controller.config.pulses[qmpulse.operation] == target_pulse
    assert target_pulse["waveforms"]["I"] in controller.config.waveforms
    assert target_pulse["waveforms"]["Q"] in controller.config.waveforms


def test_qm_register_flux_pulse(qmplatform):
    qubit = 2
    platform = qmplatform
    controller = platform.instruments["qm"]
    pulse = FluxPulse(
        0, 30, 0.005, Rectangular(), platform.qubits[qubit].flux.name, qubit
    )
    target_pulse = {
        "operation": "control",
        "length": pulse.duration,
        "waveforms": {"single": "constant_wf0.005"},
    }
    qmpulse = QMPulse(pulse)
    controller.config.register_element(platform.qubits[qubit], pulse)
    controller.config.register_pulse(platform.qubits[qubit], qmpulse)
    assert controller.config.pulses[qmpulse.operation] == target_pulse
    assert target_pulse["waveforms"]["single"] in controller.config.waveforms


@pytest.mark.parametrize("duration", [0, 30])
def test_qm_register_baked_pulse(qmplatform, duration):
    platform = qmplatform
    qubit = platform.qubits[3]
    controller = platform.instruments["qm"]
    controller.config.register_flux_element(qubit)
    pulse = FluxPulse(
        3, duration, 0.05, Rectangular(), qubit.flux.name, qubit=qubit.name
    )
    qmpulse = BakedPulse(pulse)
    config = controller.config
    qmpulse.bake(config, [pulse.duration])

    assert config.elements["flux3"]["operations"] == {
        "baked_Op_0": "flux3_baked_pulse_0"
    }
    if duration == 0:
        assert config.pulses["flux3_baked_pulse_0"] == {
            "operation": "control",
            "length": 16,
            "waveforms": {"single": "flux3_baked_wf_0"},
        }
        assert config.waveforms["flux3_baked_wf_0"] == {
            "type": "arbitrary",
            "samples": 16 * [0],
            "is_overridable": False,
        }
    else:
        assert config.pulses["flux3_baked_pulse_0"] == {
            "operation": "control",
            "length": 32,
            "waveforms": {"single": "flux3_baked_wf_0"},
        }
        assert config.waveforms["flux3_baked_wf_0"] == {
            "type": "arbitrary",
            "samples": 30 * [0.05] + 2 * [0],
            "is_overridable": False,
        }


@patch("qibolab.instruments.qm.QMController.execute_program")
def test_qm_qubit_spectroscopy(mocker, qmplatform):
    platform = qmplatform
    controller = platform.instruments["qm"]
    # disable program dump otherwise it will fail if we don't connect
    controller.script_file_name = None
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    for qubit in [1, 2, 3]:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=500
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.append(qd_pulses[qubit])
        sequence.append(ro_pulses[qubit])
    options = ExecutionParameters(nshots=1024, relaxation_time=100000)
    result = controller.play(platform.qubits, platform.couplers, sequence, options)


@patch("qibolab.instruments.qm.QMController.execute_program")
def test_qm_duration_sweeper(mocker, qmplatform):
    platform = qmplatform
    controller = platform.instruments["qm"]
    # disable program dump otherwise it will fail if we don't connect
    controller.script_file_name = None
    qubit = 1
    sequence = PulseSequence()
    qd_pulse = platform.create_RX_pulse(qubit, start=0)
    sequence.append(qd_pulse)
    sequence.append(platform.create_MZ_pulse(qubit, start=qd_pulse.finish))
    sweeper = Sweeper(Parameter.duration, np.arange(2, 12, 2), pulses=[qd_pulse])
    options = ExecutionParameters(nshots=1024, relaxation_time=100000)
    if platform.name == "qm":
        result = controller.sweep(
            platform.qubits, platform.couplers, sequence, options, sweeper
        )
    else:
        with pytest.raises(ValueError):
            # TODO: Figure what is wrong with baking and Octaves
            result = controller.sweep(
                platform.qubits, platform.couplers, sequence, options, sweeper
            )
