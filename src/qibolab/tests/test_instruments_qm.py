import json
import pathlib

import numpy as np
import pytest
from qm import qua

from qibolab.instruments.qm import QMOPX, QMPulse, QMSequence
from qibolab.paths import qibolab_folder
from qibolab.platform import create_tii_qw5q_gold
from qibolab.pulses import Pulse, ReadoutPulse, Rectangular

RUNCARD = qibolab_folder / "runcards" / "qw5q_gold.yml"
DUMMY_ADDRESS = "0.0.0.0:0"
REGRESSION_FOLDER = pathlib.Path(__file__).with_name("qmregressions")


def assert_json_fixture(data, filename):
    filename = REGRESSION_FOLDER / filename
    try:
        with open(filename) as file:
            target = json.load(file)
    except:  # pragma: no cover
        # case not tested in GitHub workflows because files exist
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        target = data
    data = json.loads(json.dumps(data))
    assert data == target


def test_qmpulse():
    pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    qmpulse = QMPulse(pulse)
    assert qmpulse.operation == pulse.serial
    assert qmpulse.relative_phase == 0


def test_qmpulse_declare_output():
    pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    qmpulse = QMPulse(pulse)
    with qua.program() as _:
        qmpulse.declare_output(0.1, 0.2)
    assert qmpulse.threshold == 0.1
    assert qmpulse.cos == np.cos(0.2)
    assert qmpulse.sin == np.sin(0.2)
    assert isinstance(qmpulse.I, qua._dsl._Variable)
    assert isinstance(qmpulse.I_st, qua._dsl._ResultSource)
    assert isinstance(qmpulse.shot, qua._dsl._Variable)
    assert isinstance(qmpulse.shots, qua._dsl._ResultSource)


def test_qmsequence():
    qd_pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    ro_pulse = ReadoutPulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch1", qubit=0)
    qmsequence = QMSequence()
    with pytest.raises(TypeError):
        qmsequence.add("test")
    qmsequence.add(qd_pulse)
    qmsequence.add(ro_pulse)
    assert len(qmsequence.pulse_to_qmpulse) == 2
    assert len(qmsequence) == 2
    assert len(qmsequence.ro_pulses) == 1


# TODO: Test connect/disconnect


def test_qmopx_setup():
    platform = create_tii_qw5q_gold(RUNCARD, simulation_duration=1000, address=DUMMY_ADDRESS)
    platform.setup()
    opx = platform.design.controller
    assert opx.time_of_flight == 280
    assert opx.relaxation_time == 50000
    assert_json_fixture(opx.config, "qmopx_setup.json")


# TODO: Test start/stop


def test_qmopx_register_analog_output_controllers():
    opx = QMOPX("test", DUMMY_ADDRESS)
    opx.register_analog_output_controllers([("con1", 1), ("con1", 2)])
    controllers = opx.config["controllers"]
    assert controllers == {"con1": {"analog_outputs": {1: {"offset": 0.0}, 2: {"offset": 0.0}}}}

    opx = QMOPX("test", DUMMY_ADDRESS)
    opx.register_analog_output_controllers([("con1", 1), ("con1", 2)], offset=0.005)
    controllers = opx.config["controllers"]
    assert controllers == {"con1": {"analog_outputs": {1: {"offset": 0.005}, 2: {"offset": 0.005}}}}

    opx = QMOPX("test", DUMMY_ADDRESS)
    filters = {"feedforward": [1, -1], "feedback": [0.95]}
    opx.register_analog_output_controllers(
        [
            ("con2", 2),
        ],
        offset=0.005,
        filter=filters,
    )
    controllers = opx.config["controllers"]
    assert controllers == {
        "con2": {"analog_outputs": {2: {"filter": {"feedback": [0.95], "feedforward": [1, -1]}, "offset": 0.005}}}
    }


def test_qmopx_register_drive_element():
    platform = create_tii_qw5q_gold(RUNCARD, simulation_duration=1000, address=DUMMY_ADDRESS)
    opx = platform.design.controller
    opx.register_drive_element(platform.qubits[0], intermediate_frequency=int(1e6))
    assert "drive0" in opx.config["elements"]
    target_element = {
        "mixInputs": {"I": ("con3", 2), "Q": ("con3", 1), "lo_frequency": 4700000000, "mixer": "mixer_drive0"},
        "intermediate_frequency": 1000000,
        "operations": {},
    }
    assert opx.config["elements"]["drive0"] == target_element
    target_mixer = [{"intermediate_frequency": 1000000, "lo_frequency": 4700000000, "correction": [1.0, 0.0, 0.0, 1.0]}]
    assert opx.config["mixers"]["mixer_drive0"] == target_mixer


def test_qmpulse_bake():
    pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    qmpulse = QMPulse(pulse)
    # TODO: Needs config
