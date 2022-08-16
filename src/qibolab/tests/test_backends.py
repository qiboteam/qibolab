# -*- coding: utf-8 -*-
import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit

from qibolab.backends import QibolabBackend
from qibolab.pulses import (
    Drag,
    Gaussian,
    Pulse,
    PulseSequence,
    ReadoutPulse,
    Rectangular,
)


@pytest.mark.parametrize("platform_name", ["tii1q", "tii5q"])  # , 'icarusq'])
def test_backend_init(platform_name):
    from qibolab.platforms.multiqubit import MultiqubitPlatform

    backend = QibolabBackend(platform_name)
    assert isinstance(backend.platform, MultiqubitPlatform)


@pytest.mark.parametrize("platform_name", ["tii1q", "tii5q"])  # , 'icarusq'])
def test_circuit_to_sequence(platform_name):
    backend = QibolabBackend(platform_name)
    circuit = Circuit(1)
    circuit.add(gates.RX(0, theta=0.1))
    circuit.add(gates.RY(0, theta=0.2))
    circuit.add(gates.M(0))

    seq = PulseSequence()
    for gate in circuit.queue:
        backend.platform.to_sequence(seq, gate)
    backend.platform.to_sequence(seq, circuit.measurement_gate)

    assert len(seq.pulses) == 5
    assert len(seq.qd_pulses) == 4
    assert len(seq.ro_pulses) == 1

    RX_pulse = backend.platform.RX_pulse(0)
    MZ_pulse = backend.platform.MZ_pulse(0, RX_pulse.duration)

    phases = [np.pi / 2, 0.1 - np.pi / 2, 0.1, 0.3 - np.pi]
    for i, (pulse, phase) in enumerate(zip(seq.pulses[:-1], phases)):
        assert pulse.channel == RX_pulse.channel
        np.testing.assert_allclose(pulse.start, i * RX_pulse.duration)
        np.testing.assert_allclose(pulse.duration, RX_pulse.duration)
        np.testing.assert_allclose(pulse.amplitude, RX_pulse.amplitude / 2)
        np.testing.assert_allclose(pulse.frequency, RX_pulse.frequency)
        np.testing.assert_allclose(pulse.phase, phase)

    pulse = seq.pulses[-1]
    start = 4 * RX_pulse.duration
    np.testing.assert_allclose(pulse.start, start)
    np.testing.assert_allclose(pulse.duration, MZ_pulse.duration)
    np.testing.assert_allclose(pulse.amplitude, MZ_pulse.amplitude)
    np.testing.assert_allclose(pulse.frequency, MZ_pulse.frequency)
    np.testing.assert_allclose(pulse.phase, 0.3)


@pytest.mark.parametrize("platform_name", ["tii1q", "tii5q"])  # , 'icarusq'])
def test_execute_circuit_errors(platform_name):
    backend = QibolabBackend(platform_name)
    circuit = Circuit(1)
    circuit.add(gates.X(0))
    with pytest.raises(RuntimeError):
        result = backend.execute_circuit(circuit)
    circuit.add(gates.M(0))
    with pytest.raises(ValueError):
        result = backend.execute_circuit(circuit, initial_state=np.ones(2))


@pytest.mark.xfail
@pytest.mark.parametrize("platform_name", ["tii1q", "tii5q"])  # , 'icarusq'])
def test_execute_circuit(platform_name):
    # TODO: Test this method on IcarusQ
    backend = QibolabBackend(platform_name)
    circuit = Circuit(1)
    circuit.add(gates.X(0))
    circuit.add(gates.M(0))
    result = backend.execute_circuit(circuit, nshots=100)
    # disconnect from instruments so that they are available for other tests
    backend.platform.disconnect()
