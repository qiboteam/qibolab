from unittest.mock import patch

import numpy as np
import pytest
from qm import qua

from qibolab import AcquisitionType, ExecutionParameters, create_platform
from qibolab.instruments.qm import QMOPX, QMPort
from qibolab.instruments.qm.acquisition import Acquisition
from qibolab.instruments.qm.sequence import BakedPulse, QMPulse, Sequence
from qibolab.pulses import FluxPulse, Pulse, PulseSequence, ReadoutPulse, Rectangular
from qibolab.sweeper import Parameter, Sweeper


def test_qmpulse():
    pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    qmpulse = QMPulse(pulse)
    assert qmpulse.operation == pulse.serial
    assert qmpulse.relative_phase == 0


@pytest.mark.parametrize("acquisition_type", AcquisitionType)
def test_qmpulse_declare_output(acquisition_type):
    options = ExecutionParameters(acquisition_type=acquisition_type)
    pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    qmpulse = QMPulse(pulse)
    if acquisition_type is AcquisitionType.SPECTROSCOPY:
        with pytest.raises(ValueError):
            with qua.program() as _:
                qmpulse.declare_output(options, 0.1, 0.2)
    else:
        with qua.program() as _:
            qmpulse.declare_output(options, 0.1, 0.2)
        acquisition = qmpulse.acquisition
        assert isinstance(acquisition, Acquisition)
        if acquisition_type is AcquisitionType.DISCRIMINATION:
            assert acquisition.threshold == 0.1
            assert acquisition.cos == np.cos(0.2)
            assert acquisition.sin == np.sin(0.2)
            assert isinstance(acquisition.shot, qua._dsl._Variable)
            assert isinstance(acquisition.shots, qua._dsl._ResultSource)
        elif acquisition_type is AcquisitionType.INTEGRATION:
            assert isinstance(acquisition.I, qua._dsl._Variable)
            assert isinstance(acquisition.Q, qua._dsl._Variable)
            assert isinstance(acquisition.I_stream, qua._dsl._ResultSource)
            assert isinstance(acquisition.Q_stream, qua._dsl._ResultSource)
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
    assert len(qmsequence.ro_pulses) == 1


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


# TODO: Test connect/disconnect


def test_qmopx_setup(dummy_qrc):
    platform = create_platform("qm")
    opx = platform.instruments["qmopx"]
    assert opx.time_of_flight == 280


def test_qmopx_register_analog_output_controllers():
    name = "test"
    address = "0.0.0.0:0"
    opx = QMOPX(name, address)
    port = QMPort((("con1", 1), ("con1", 2)))
    opx.config.register_analog_output_controllers(port)
    controllers = opx.config.controllers
    assert controllers == {
        "con1": {"analog_outputs": {1: {"offset": 0.0}, 2: {"offset": 0.0}}}
    }

    opx = QMOPX(name, address)
    port = QMPort((("con1", 1), ("con1", 2)))
    port.offset = 0.005
    opx.config.register_analog_output_controllers(port)
    controllers = opx.config.controllers
    assert controllers == {
        "con1": {"analog_outputs": {1: {"offset": 0.005}, 2: {"offset": 0.005}}}
    }

    opx = QMOPX(name, address)
    port = QMPort((("con2", 2),))
    port.offset = 0.005
    port.filters = {"feedforward": [1, -1], "feedback": [0.95]}
    opx.config.register_analog_output_controllers(port)
    controllers = opx.config.controllers
    assert controllers == {
        "con2": {
            "analog_outputs": {
                2: {
                    "filter": {"feedback": [0.95], "feedforward": [1, -1]},
                    "offset": 0.005,
                }
            }
        }
    }


def test_qmopx_register_drive_element(dummy_qrc):
    platform = create_platform("qm")
    opx = platform.instruments["qmopx"]
    opx.config.register_drive_element(
        platform.qubits[0], intermediate_frequency=int(1e6)
    )
    assert "drive0" in opx.config.elements
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
    assert opx.config.elements["drive0"] == target_element
    target_mixer = [
        {
            "intermediate_frequency": 1000000,
            "lo_frequency": 4700000000,
            "correction": [1.0, 0.0, 0.0, 1.0],
        }
    ]
    assert opx.config.mixers["mixer_drive0"] == target_mixer


def test_qmopx_register_readout_element(dummy_qrc):
    platform = create_platform("qm")
    opx = platform.instruments["qmopx"]
    opx.config.register_readout_element(
        platform.qubits[2], int(1e6), opx.time_of_flight, opx.smearing
    )
    assert "readout2" in opx.config.elements
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
    assert opx.config.elements["readout2"] == target_element
    target_mixer = [
        {
            "intermediate_frequency": 1000000,
            "lo_frequency": 7900000000,
            "correction": [1.0, 0.0, 0.0, 1.0],
        }
    ]
    assert opx.config.mixers["mixer_readout2"] == target_mixer


@pytest.mark.parametrize("pulse_type,qubit", [("drive", 2), ("readout", 1)])
def test_qmopx_register_pulse(dummy_qrc, pulse_type, qubit):
    platform = create_platform("qm")
    opx = platform.instruments["qmopx"]
    if pulse_type == "drive":
        pulse = platform.create_RX_pulse(qubit, start=0)
        target_pulse = {
            "operation": "control",
            "length": pulse.duration,
            "waveforms": {
                "I": pulse.envelope_waveform_i().serial,
                "Q": pulse.envelope_waveform_q().serial,
            },
        }

    else:
        pulse = platform.create_MZ_pulse(qubit, start=0)
        target_pulse = {
            "operation": "measurement",
            "length": pulse.duration,
            "waveforms": {"I": "constant_wf0.003575", "Q": "constant_wf0.003575"},
            "digital_marker": "ON",
            "integration_weights": {
                "cos": "cosine_weights1",
                "minus_sin": "minus_sine_weights1",
                "sin": "sine_weights1",
            },
        }

    opx.config.register_element(
        platform.qubits[qubit], pulse, opx.time_of_flight, opx.smearing
    )
    opx.config.register_pulse(platform.qubits[qubit], pulse)
    assert opx.config.pulses[pulse.serial] == target_pulse
    assert target_pulse["waveforms"]["I"] in opx.config.waveforms
    assert target_pulse["waveforms"]["Q"] in opx.config.waveforms
    assert (
        opx.config.elements[f"{pulse_type}{qubit}"]["operations"][pulse.serial]
        == pulse.serial
    )


def test_qmopx_register_flux_pulse(dummy_qrc):
    qubit = 2
    platform = create_platform("qm")
    opx = platform.instruments["qmopx"]
    pulse = FluxPulse(
        0, 30, 0.005, Rectangular(), platform.qubits[qubit].flux.name, qubit
    )
    target_pulse = {
        "operation": "control",
        "length": pulse.duration,
        "waveforms": {"single": "constant_wf0.005"},
    }
    opx.config.register_element(platform.qubits[qubit], pulse)
    opx.config.register_pulse(platform.qubits[qubit], pulse)
    assert opx.config.pulses[pulse.serial] == target_pulse
    assert target_pulse["waveforms"]["single"] in opx.config.waveforms
    assert (
        opx.config.elements[f"flux{qubit}"]["operations"][pulse.serial] == pulse.serial
    )


@pytest.mark.parametrize("duration", [0, 30])
def test_qmopx_register_baked_pulse(dummy_qrc, duration):
    platform = create_platform("qm")
    qubit = platform.qubits[3]
    opx = platform.instruments["qmopx"]
    opx.config.register_flux_element(qubit)
    pulse = FluxPulse(
        3, duration, 0.05, Rectangular(), qubit.flux.name, qubit=qubit.name
    )
    qmpulse = BakedPulse(pulse)
    config = opx.config
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


@patch("qibolab.instruments.qm.simulator.QMSim.execute_program")
def test_qmopx_qubit_spectroscopy(mocker):
    platform = create_platform("qm")
    opx = platform.instruments["qmopx"]
    # disable program dump otherwise it will fail if we don't connect
    opx.script_file_name = None
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
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])
    options = ExecutionParameters(nshots=1024, relaxation_time=100000)
    result = opx.play(platform.qubits, platform.couplers, sequence, options)


@patch("qibolab.instruments.qm.simulator.QMSim.execute_program")
def test_qmopx_duration_sweeper(mocker):
    platform = create_platform("qm")
    opx = platform.instruments["qmopx"]
    # disable program dump otherwise it will fail if we don't connect
    opx.script_file_name = None
    qubit = 1
    sequence = PulseSequence()
    qd_pulse = platform.create_RX_pulse(qubit, start=0)
    sequence.add(qd_pulse)
    sequence.add(platform.create_MZ_pulse(qubit, start=qd_pulse.finish))
    sweeper = Sweeper(Parameter.duration, np.arange(2, 12, 2), pulses=[qd_pulse])
    options = ExecutionParameters(nshots=1024, relaxation_time=100000)
    result = opx.sweep(platform.qubits, platform.couplers, sequence, options, sweeper)
