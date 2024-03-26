"""Test compilation of different pulse sequences using the Quantum Machines
simulator.

In order to run these tests, provide the following options through the ``pytest`` parser:
    address (str): token for the QM simulator
    simulation-duration (int): Duration for the simulation in ns.
    folder (str): Optional folder to save the generated waveforms for each test.
If a folder is provided the waveforms will be generated and saved during the first run.
For every other run, the generated waveforms will be compared with the saved ones and errors
will be raised if there is disagreement.
If an error is raised or a waveform is generated for the first time, a plot will also be
created so that the user can check if the waveform looks as expected.
"""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters, create_platform
from qibolab.backends import QibolabBackend
from qibolab.pulses import SNZ, Pulse, PulseSequence, Rectangular
from qibolab.sweeper import Parameter, Sweeper

from .conftest import set_platform_profile


@pytest.fixture(scope="module")
def simulator(request):
    """Platform using the QM cloud simulator.

    Requires the address for connecting to the simulator, which is
    provided via command line. If an address is not provided these tests
    are skipped.
    """
    set_platform_profile()
    address = request.config.getoption("--address")
    if address is None:
        pytest.skip("Skipping QM simulator tests because address was not provided.")

    platform = create_platform("qm")
    controller = platform.instruments["qm"]
    controller.simulation_duration = request.config.getoption("--simulation-duration")
    controller.time_of_flight = 280
    # controller.cloud = True

    platform.connect()
    yield platform
    platform.disconnect()


@pytest.fixture(scope="module")
def folder(request):
    return request.config.getoption("--folder")


def assert_regression(samples, folder=None, filename=None):
    """Assert that simulated data agree with the saved regression.

    If a regression does not exist it is created and the corresponding
    waveforms are plotted, so that the user can confirm that they look
    as expected.

    Args:
        samples (dict): Dictionary holding the waveforms as returned by the QM simulator.
        filename (str): Name of the file that contains the regressions to compare with.
    """

    def plot():
        plt.figure()
        plt.title(filename)
        for con in ["con1", "con2", "con3"]:
            if hasattr(samples, con):
                sample = getattr(samples, con)
                sample.plot()
        plt.show()

    if folder is None:
        plot()
    else:
        path = os.path.join(folder, f"{filename}.hdf5")
        if os.path.exists(path):
            file = h5py.File(path, "r")
            for con, target_data in file.items():
                sample = getattr(samples, con)
                for port, target_waveform in target_data.items():
                    waveform = sample.analog[port]
                    try:
                        np.testing.assert_allclose(waveform, target_waveform[:])
                    except AssertionError as exception:
                        np.savetxt(os.path.join(folder, "waveform.txt"), waveform)
                        np.savetxt(
                            os.path.join(folder, "target_waveform.txt"),
                            target_waveform[:],
                        )
                        plot()
                        raise exception

        else:
            plot()
            if not os.path.exists(folder):
                os.mkdir(folder)
            file = h5py.File(path, "w")
            # TODO: Generalize for arbitrary number of controllers
            for con in ["con1", "con2", "con3"]:
                if hasattr(samples, con):
                    sample = getattr(samples, con)
                    group = file.create_group(con)
                    for port, waveform in sample.analog.items():
                        group.create_dataset(port, data=waveform, compression="gzip")


def test_qmsim_resonator_spectroscopy(simulator, folder):
    qubits = list(range(simulator.nqubits))
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = simulator.create_qubit_readout_pulse(qubit, start=0)
        sequence.append(ro_pulses[qubit])
    options = ExecutionParameters(nshots=1)
    result = simulator.execute_pulse_sequence(sequence, options)
    samples = result.get_simulated_samples()
    assert_regression(samples, folder, "resonator_spectroscopy")


def test_qmsim_qubit_spectroscopy(simulator, folder):
    qubits = list(range(simulator.nqubits))
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = simulator.create_qubit_drive_pulse(
            qubit, start=0, duration=500
        )
        qd_pulses[qubit].amplitude = 0.05
        ro_pulses[qubit] = simulator.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.append(qd_pulses[qubit])
        sequence.append(ro_pulses[qubit])
    options = ExecutionParameters(nshots=1)
    result = simulator.execute_pulse_sequence(sequence, options)
    samples = result.get_simulated_samples()
    assert_regression(samples, folder, "qubit_spectroscopy")


@pytest.mark.parametrize(
    "parameter,values",
    [
        (Parameter.frequency, np.array([0, 1e6])),
        (Parameter.amplitude, np.array([0.5, 1.0])),
        (Parameter.relative_phase, np.array([0, 1.0])),
    ],
)
def test_qmsim_sweep(simulator, folder, parameter, values):
    qubits = list(range(simulator.nqubits))
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = simulator.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = simulator.create_MZ_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.append(qd_pulses[qubit])
        sequence.append(ro_pulses[qubit])
    pulses = [qd_pulses[qubit] for qubit in qubits]
    sweeper = Sweeper(parameter, values, pulses)
    options = ExecutionParameters(
        nshots=1,
        relaxation_time=20,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    result = simulator.sweep(sequence, options, sweeper)
    samples = result.get_simulated_samples()
    assert_regression(samples, folder, f"sweep_{parameter.name}")


def test_qmsim_sweep_bias(simulator, folder):
    qubits = list(range(simulator.nqubits))
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = simulator.create_MZ_pulse(qubit, start=0)
        sequence.append(ro_pulses[qubit])
    values = [0, 0.005]
    sweeper = Sweeper(
        Parameter.bias, values, qubits=[simulator.qubits[q] for q in qubits]
    )
    options = ExecutionParameters(
        nshots=1,
        relaxation_time=20,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    result = simulator.sweep(sequence, options, sweeper)
    samples = result.get_simulated_samples()
    assert_regression(samples, folder, "sweep_bias")


def test_qmsim_sweep_start(simulator, folder):
    qubits = list(range(simulator.nqubits))
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = simulator.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = simulator.create_MZ_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.append(qd_pulses[qubit])
        sequence.append(ro_pulses[qubit])
    values = [20, 40]
    pulses = [ro_pulses[qubit] for qubit in qubits]
    sweeper = Sweeper(Parameter.start, values, pulses=pulses)
    options = ExecutionParameters(
        nshots=1,
        relaxation_time=0,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    result = simulator.sweep(sequence, options, sweeper)
    samples = result.get_simulated_samples()
    assert_regression(samples, folder, "sweep_start")


def test_qmsim_sweep_start_two_pulses(simulator, folder):
    qubits = list(range(simulator.nqubits))
    sequence = PulseSequence()
    qd_pulses1 = {}
    qd_pulses2 = {}
    ro_pulses = {}
    for qubit in qubits:
        qd_pulses1[qubit] = simulator.create_RX_pulse(qubit, start=0)
        qd_pulses2[qubit] = simulator.create_RX_pulse(
            qubit, start=qd_pulses1[qubit].finish
        )
        ro_pulses[qubit] = simulator.create_MZ_pulse(
            qubit, start=qd_pulses2[qubit].finish
        )
        sequence.append(qd_pulses1[qubit])
        sequence.append(qd_pulses2[qubit])
        sequence.append(ro_pulses[qubit])
    values = [20, 60]
    pulses = [qd_pulses2[qubit] for qubit in qubits]
    sweeper = Sweeper(Parameter.start, values, pulses=pulses)
    options = ExecutionParameters(
        nshots=1,
        relaxation_time=0,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    result = simulator.sweep(sequence, options, sweeper)
    samples = result.get_simulated_samples()
    assert_regression(samples, folder, "sweep_start_two_pulses")


def test_qmsim_sweep_duration(simulator, folder):
    controller = simulator.instruments["qmopx"]
    original_duration = controller.simulation_duration
    controller.simulation_duration = 1250
    qubits = list(range(simulator.nqubits))
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = simulator.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = simulator.create_MZ_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.append(qd_pulses[qubit])
        sequence.append(ro_pulses[qubit])
    values = [20, 60]
    pulses = [qd_pulses[qubit] for qubit in qubits]
    sweeper = Sweeper(Parameter.duration, values, pulses=pulses)
    options = ExecutionParameters(
        nshots=1,
        relaxation_time=0,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    result = simulator.sweep(sequence, options, sweeper)
    samples = result.get_simulated_samples()
    assert_regression(samples, folder, "sweep_duration")
    controller.simulation_duration = original_duration


def test_qmsim_sweep_duration_two_pulses(simulator, folder):
    controller = simulator.instruments["qmopx"]
    original_duration = controller.simulation_duration
    controller.simulation_duration = 1250
    qubits = list(range(simulator.nqubits))
    sequence = PulseSequence()
    qd_pulses1 = {}
    qd_pulses2 = {}
    ro_pulses = {}
    for qubit in qubits:
        qd_pulses1[qubit] = simulator.create_RX_pulse(qubit, start=0)
        qd_pulses2[qubit] = simulator.create_RX_pulse(
            qubit, start=qd_pulses1[qubit].finish
        )
        ro_pulses[qubit] = simulator.create_MZ_pulse(
            qubit, start=qd_pulses2[qubit].finish
        )
        sequence.append(qd_pulses1[qubit])
        sequence.append(qd_pulses2[qubit])
        sequence.append(ro_pulses[qubit])
    values = [20, 60]
    pulses = [qd_pulses1[qubit] for qubit in qubits]
    sweeper = Sweeper(Parameter.duration, values, pulses=pulses)
    options = ExecutionParameters(
        nshots=1,
        relaxation_time=0,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    result = simulator.sweep(sequence, options, sweeper)
    samples = result.get_simulated_samples()
    assert_regression(samples, folder, "sweep_duration_two_pulses")
    controller.simulation_duration = original_duration


gatelist = [
    ["I", "I"],
    ["RX(pi)", "RX(pi)"],
    ["RY(pi)", "RY(pi)"],
    ["RX(pi)", "RY(pi)"],
    ["RY(pi)", "RX(pi)"],
    ["RX(pi/2)", "I"],
    ["RY(pi/2)", "I"],
    ["RX(pi/2)", "RY(pi/2)"],
    ["RY(pi/2)", "RX(pi/2)"],
    ["RX(pi/2)", "RY(pi)"],
    ["RY(pi/2)", "RX(pi)"],
    ["RX(pi)", "RY(pi/2)"],
    ["RY(pi)", "RX(pi/2)"],
    ["RX(pi/2)", "RX(pi)"],
    ["RX(pi)", "RX(pi/2)"],
    ["RY(pi/2)", "RY(pi)"],
    ["RY(pi)", "RY(pi/2)"],
    ["RX(pi)", "I"],
    ["RY(pi)", "I"],
    ["RX(pi/2)", "RX(pi/2)"],
    ["RY(pi/2)", "RY(pi/2)"],
]


@pytest.mark.parametrize("count,gate_pair", enumerate(gatelist))
def test_qmsim_allxy(simulator, folder, count, gate_pair):
    qubits = [1, 2, 3, 4]
    allxy_pulses = {
        "I": lambda qubit, start: None,
        "RX(pi)": lambda qubit, start: simulator.create_RX_pulse(qubit, start=start),
        "RX(pi/2)": lambda qubit, start: simulator.create_RX90_pulse(
            qubit, start=start
        ),
        "RY(pi)": lambda qubit, start: simulator.create_RX_pulse(
            qubit, start=start, relative_phase=np.pi / 2
        ),
        "RY(pi/2)": lambda qubit, start: simulator.create_RX90_pulse(
            qubit, start=start, relative_phase=np.pi / 2
        ),
    }

    sequence = PulseSequence()
    for qubit in qubits:
        start = 0
        for gate in gate_pair:
            pulse = allxy_pulses[gate](qubit, start)
            if pulse is not None:
                sequence.append(pulse)
                start += pulse.duration
        sequence.append(simulator.create_MZ_pulse(qubit, start=start))

    options = ExecutionParameters(nshots=1)
    result = simulator.execute_pulse_sequence(sequence, options)
    samples = result.get_simulated_samples()
    assert_regression(samples, folder, f"allxy{count}")


@pytest.mark.parametrize("sweep", [None, "1D", "2D"])
def test_qmsim_chevron(simulator, folder, sweep):
    lowfreq, highfreq = 1, 2
    initialize_1 = simulator.create_RX_pulse(lowfreq, start=0, relative_phase=0)
    initialize_2 = simulator.create_RX_pulse(highfreq, start=0, relative_phase=0)
    flux_pulse = Pulse.flux(
        start=initialize_2.finish,
        duration=31,
        amplitude=0.05,
        shape=Rectangular(),
        channel=simulator.qubits[highfreq].flux.name,
        qubit=highfreq,
    )
    measure_lowfreq = simulator.create_qubit_readout_pulse(
        lowfreq, start=flux_pulse.finish
    )
    measure_highfreq = simulator.create_qubit_readout_pulse(
        highfreq, start=flux_pulse.finish
    )
    sequence = PulseSequence()
    sequence.append(initialize_1)
    sequence.append(initialize_2)
    sequence.append(flux_pulse)
    sequence.append(measure_lowfreq)
    sequence.append(measure_highfreq)

    options = ExecutionParameters(
        nshots=1,
        relaxation_time=0,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    if sweep is None:
        result = simulator.execute_pulse_sequence(sequence, options)
    elif sweep == "1D":
        sweeper = Sweeper(Parameter.duration, [10, 60], pulses=[flux_pulse])
        result = simulator.sweep(sequence, options, sweeper)
    elif sweep == "2D":
        duration_sweeper = Sweeper(Parameter.duration, [10, 40], pulses=[flux_pulse])
        amplitude_sweeper = Sweeper(Parameter.amplitude, [0.5, 2], pulses=[flux_pulse])
        result = simulator.sweep(sequence, options, duration_sweeper, amplitude_sweeper)
    samples = result.get_simulated_samples()
    if sweep is None:
        assert_regression(samples, folder, "chevron")
    else:
        assert_regression(samples, folder, f"chevron_sweep_{sweep}")


@pytest.mark.parametrize("qubits", [[1, 2], [2, 3]])
@pytest.mark.parametrize("use_flux_pulse", [True, False])
def test_qmsim_tune_landscape(simulator, folder, qubits, use_flux_pulse):
    lowfreq, highfreq = min(qubits), max(qubits)

    y90_pulse = simulator.create_RX90_pulse(lowfreq, start=0, relative_phase=np.pi / 2)
    x_pulse_start = simulator.create_RX_pulse(highfreq, start=0, relative_phase=0)
    if use_flux_pulse:
        flux_pulse = Pulse.flux(
            start=y90_pulse.finish,
            duration=30,
            amplitude=0.055,
            shape=Rectangular(),
            channel=simulator.qubits[highfreq].flux.name,
            qubit=highfreq,
        )
        theta_pulse = simulator.create_RX90_pulse(
            lowfreq, start=flux_pulse.finish, relative_phase=np.pi / 3
        )
        x_pulse_end = simulator.create_RX_pulse(
            highfreq, start=flux_pulse.finish, relative_phase=0
        )
    else:
        theta_pulse = simulator.create_RX90_pulse(
            lowfreq, start=y90_pulse.finish, relative_phase=np.pi / 3
        )
        x_pulse_end = simulator.create_RX_pulse(
            highfreq, start=x_pulse_start.finish, relative_phase=0
        )

    measure_lowfreq = simulator.create_qubit_readout_pulse(
        lowfreq, start=theta_pulse.finish
    )
    measure_highfreq = simulator.create_qubit_readout_pulse(
        highfreq, start=x_pulse_end.finish
    )

    sequence = x_pulse_start + y90_pulse
    if use_flux_pulse:
        sequence += flux_pulse
    sequence += theta_pulse + x_pulse_end
    sequence += measure_lowfreq + measure_highfreq

    options = ExecutionParameters(nshots=1)
    result = simulator.execute_pulse_sequence(sequence, options)
    samples = result.get_simulated_samples()
    qubitstr = "".join(str(q) for q in qubits)
    if use_flux_pulse:
        assert_regression(samples, folder, f"tune_landscape_{qubitstr}")
    else:
        assert_regression(samples, folder, f"tune_landscape_noflux_{qubitstr}")


@pytest.mark.parametrize("qubit", [2, 3])
def test_qmsim_snz_pulse(simulator, folder, qubit):
    duration = 30
    amplitude = 0.01
    sequence = PulseSequence()
    shape = SNZ(t_half_flux_pulse=duration // 2, b_amplitude=2)
    channel = simulator.qubits[qubit].flux.name
    qd_pulse = simulator.create_RX_pulse(qubit, start=0)
    flux_pulse = Pulse.flux(qd_pulse.finish, duration, amplitude, shape, channel, qubit)
    ro_pulse = simulator.create_MZ_pulse(qubit, start=flux_pulse.finish)
    sequence.append(qd_pulse)
    sequence.append(flux_pulse)
    sequence.append(ro_pulse)
    options = ExecutionParameters(nshots=1)
    result = simulator.execute_pulse_sequence(sequence, options)
    samples = result.get_simulated_samples()
    assert_regression(samples, folder, f"snz_pulse_{qubit}")


@pytest.mark.parametrize("qubits", [[1, 2], [2, 3]])
def test_qmsim_bell_circuit(simulator, folder, qubits):
    backend = QibolabBackend(simulator)
    circuit = Circuit(5)
    circuit.append(gates.H(qubits[0]))
    circuit.append(gates.CNOT(*qubits))
    circuit.append(gates.M(*qubits))
    result = backend.execute_circuit(circuit, nshots=1)
    result = result.execution_result
    samples = result.get_simulated_samples()
    qubitstr = "".join(str(q) for q in qubits)
    assert_regression(samples, folder, f"bell_circuit_{qubitstr}")


def test_qmsim_ghz_circuit(simulator, folder):
    backend = QibolabBackend(simulator)
    circuit = Circuit(5)
    circuit.append(gates.H(2))
    circuit.append(gates.CNOT(2, 1))
    circuit.append(gates.CNOT(2, 3))
    circuit.append(gates.M(1, 2, 3))
    result = backend.execute_circuit(circuit, nshots=1)
    result = result.execution_result
    samples = result.get_simulated_samples()
    assert_regression(samples, folder, "ghz_circuit_123")
