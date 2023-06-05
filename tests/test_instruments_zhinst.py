import laboneq.simple as lo
import numpy as np
import pytest

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters, create_platform
from qibolab.instruments.zhinst import ZhPulse, ZhSweeperLine
from qibolab.pulses import FluxPulse, PulseSequence, Rectangular
from qibolab.sweeper import Parameter, Sweeper


@pytest.mark.qpu
def test_connections():
    platform = create_platform("iqm5q")
    IQM5q = platform.instruments[0]
    IQM5q.start()
    IQM5q.stop()
    IQM5q.disconnect()
    IQM5q.connect()


@pytest.mark.qpu
def test_experiment_execute_pulse_sequence():
    platform = create_platform("iqm5q")
    platform.connect()
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
def test_experiment_sweep_2d_specific():
    platform = create_platform("iqm5q")
    platform.connect()
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
