from unittest.mock import Mock

import numpy as np
import pytest

from qibolab import AveragingMode, ExecutionParameters
from qibolab.instruments.qblox.controller import SEQUENCER_MEMORY, QbloxController
from qibolab.pulses import Gaussian, Pulse, PulseSequence, PulseType, Rectangular
from qibolab.result import IntegratedResults
from qibolab.sweeper import Parameter, Sweeper

from .qblox_fixtures import connected_controller, controller


def test_init(controller: QbloxController):
    assert controller.is_connected is False
    assert type(controller.modules) == dict
    assert controller.cluster == None
    assert controller._reference_clock in ["internal", "external"]


def test_sweep_too_many_bins(platform, controller):
    """Sweeps that require more bins than the hardware supports should be split
    and executed."""
    qubit = platform.qubits[0]
    pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Gaussian(5), qubit.drive.name, qubit=0)
    ro_pulse = Pulse(
        0,
        40,
        0.05,
        int(3e9),
        0.0,
        Rectangular(),
        qubit.readout.name,
        PulseType.READOUT,
        qubit=0,
    )
    sequence = PulseSequence([pulse, ro_pulse])

    # These values shall result into execution in two rounds
    shots = 128
    sweep_len = (SEQUENCER_MEMORY + 431) // shots

    mock_data = np.array([1, 2, 3, 4])
    sweep_ampl = Sweeper(Parameter.amplitude, np.random.rand(sweep_len), pulses=[pulse])
    params = ExecutionParameters(
        nshots=shots, relaxation_time=10, averaging_mode=AveragingMode.SINGLESHOT
    )
    controller._execute_pulse_sequence = Mock(
        return_value={ro_pulse.id: IntegratedResults(mock_data)}
    )
    res = controller.sweep(
        {0: platform.qubits[0]}, platform.couplers, sequence, params, sweep_ampl
    )
    expected_data = np.append(mock_data, mock_data)  #
    assert np.array_equal(res[ro_pulse.id].voltage, expected_data)


def test_sweep_too_many_sweep_points(platform, controller):
    """Sweeps that require too many bins because simply the number of sweep
    points is too large should be rejected."""
    qubit = platform.qubits[0]
    pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Gaussian(5), qubit.drive.name, qubit=0)
    sweep = Sweeper(
        Parameter.amplitude, np.random.rand(SEQUENCER_MEMORY + 17), pulses=[pulse]
    )
    params = ExecutionParameters(nshots=12, relaxation_time=10)
    with pytest.raises(ValueError, match="total number of sweep points"):
        controller.sweep({0: qubit}, {}, PulseSequence([pulse]), params, sweep)


@pytest.mark.qpu
def connect(connected_controller: QbloxController):
    connected_controller.connect()
    assert connected_controller.is_connected
    for module in connected_controller.modules.values():
        assert module.is_connected


@pytest.mark.qpu
def disconnect(connected_controller: QbloxController):
    connected_controller.connect()
    connected_controller.disconnect()
    assert connected_controller.is_connected is False
    for module in connected_controller.modules.values():
        assert module.is_connected is False
