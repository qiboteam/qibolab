"""Test compilation of different pulse sequences using the Quantum Machines simulator."""
import os
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit

from qibolab.backends import QibolabBackend
from qibolab.instruments.qm import QMOPX, QMPulse, QMSequence
from qibolab.paths import qibolab_folder
from qibolab.platform import create_tii_qw5q_gold
from qibolab.pulses import FluxPulse, Pulse, PulseSequence, ReadoutPulse, Rectangular

REGRESSION_FOLDER = pathlib.Path(__file__).with_name("regressions")
# WARNING: changing the following parameters will break all saved regressions
SIMULATION_DURATION = 3000
RUNCARD = qibolab_folder / "runcards" / "qw5q_gold.yml"


@pytest.fixture
def simulator(address):
    """Platform using the QM cloud simulator.

    Args:
        address (str): Address for connecting to the simulator. Provided via command line.
    """
    platform = create_tii_qw5q_gold(RUNCARD, simulation_duration=SIMULATION_DURATION, address=address, cloud=True)
    platform.connect()
    platform.setup()
    yield platform
    platform.disconnect()


def assert_regression(samples, filename):
    """Assert that simulated data agree with the saved regression.

    If a regression does not exist it is created and the corresponding
    waveforms are plotted, so that the user can confirm that they look
    as expected.

    Args:
        samples (dict): Dictionary holding the waveforms as returned by the QM simulator.
        filename (str): Name of the file that contains the regressions to compare with.
    """
    path = REGRESSION_FOLDER / f"{filename}.hdf5"

    def plot():
        plt.figure()
        for con in ["con1", "con2", "con3"]:
            if hasattr(samples, con):
                sample = getattr(samples, con)
                sample.plot()
        # plt.savefig(REGRESSION_FOLDER / f"{filename}.png")
        plt.show()

    if os.path.exists(path):
        file = h5py.File(path, "r")
        for con, target_data in file.items():
            sample = getattr(samples, con)
            for port, target_waveform in target_data.items():
                waveform = sample.analog[port]
                try:
                    np.testing.assert_allclose(waveform, target_waveform[:])
                except AssertionError as exception:
                    plot()
                    raise exception

    else:
        file = h5py.File(path, "w")
        # TODO: Generalize for arbitrary number of controllers
        for con in ["con1", "con2", "con3"]:
            if hasattr(samples, con):
                sample = getattr(samples, con)
                group = file.create_group(con)
                for port, waveforms in sample.analog.items():
                    group[port] = waveforms
        plot()


def test_qmsim_resonator_spectroscopy(simulator):
    qubits = list(range(simulator.nqubits))
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = simulator.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])
    result = simulator.execute_pulse_sequence(sequence, nshots=1)
    samples = result.get_simulated_samples()
    assert_regression(samples, "resonator_spectroscopy")


def test_qmsim_qubit_spectroscopy(simulator):
    qubits = list(range(simulator.nqubits))
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = simulator.create_qubit_drive_pulse(qubit, start=0, duration=500)
        qd_pulses[qubit].amplitude = 0.05
        ro_pulses[qubit] = simulator.create_qubit_readout_pulse(qubit, start=qd_pulses[qubit].finish)
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])
    result = simulator.execute_pulse_sequence(sequence, nshots=1)
    samples = result.get_simulated_samples()
    assert_regression(samples, "qubit_spectroscopy")


@pytest.mark.parametrize("qubits", [[1, 2], [2, 3]])
def test_qmsim_bell_circuit(simulator, qubits):
    backend = QibolabBackend(simulator)
    circuit = Circuit(5)
    circuit.add(gates.H(qubits[0]))
    circuit.add(gates.CNOT(*qubits))
    circuit.add(gates.M(*qubits))
    result = backend.execute_circuit(circuit, nshots=1, check_transpiled=True)
    result = result.execution_result
    samples = result.get_simulated_samples()
    qubitstr = "".join(str(q) for q in qubits)
    assert_regression(samples, f"bell_circuit_{qubitstr}")


@pytest.mark.parametrize("qubits", [[1, 2]])
@pytest.mark.parametrize("use_flux_pulse", [True, False])
def test_qmsim_tune_landscape(simulator, qubits, use_flux_pulse):
    lowfreq, highfreq = min(qubits), max(qubits)

    y90_pulse = simulator.create_RX90_pulse(lowfreq, start=0, relative_phase=np.pi / 2)
    x_pulse_start = simulator.create_RX_pulse(highfreq, start=0, relative_phase=0)
    if use_flux_pulse:
        flux_pulse = FluxPulse(
            start=y90_pulse.se_finish,
            duration=30,
            amplitude=0.055,
            shape=Rectangular(),
            channel=simulator.qubits[highfreq].flux.name,
            qubit=highfreq,
        )
        theta_pulse = simulator.create_RX90_pulse(lowfreq, start=flux_pulse.se_finish, relative_phase=np.pi / 3)
        x_pulse_end = simulator.create_RX_pulse(highfreq, start=flux_pulse.se_finish, relative_phase=0)
    else:
        theta_pulse = simulator.create_RX90_pulse(lowfreq, start=y90_pulse.se_finish, relative_phase=np.pi / 3)
        x_pulse_end = simulator.create_RX_pulse(highfreq, start=x_pulse_start.se_finish, relative_phase=0)

    measure_lowfreq = simulator.create_qubit_readout_pulse(lowfreq, start=theta_pulse.se_finish)
    measure_highfreq = simulator.create_qubit_readout_pulse(highfreq, start=x_pulse_end.se_finish)

    sequence = x_pulse_start + y90_pulse
    if use_flux_pulse:
        sequence += flux_pulse
    sequence += theta_pulse + x_pulse_end
    sequence += measure_lowfreq + measure_highfreq

    result = simulator.execute_pulse_sequence(sequence, nshots=1)
    samples = result.get_simulated_samples()
    qubitstr = "".join(str(q) for q in qubits)
    if use_flux_pulse:
        assert_regression(samples, f"tune_landscape_{qubitstr}")
    else:
        assert_regression(samples, f"tune_landscape_noflux_{qubitstr}")
