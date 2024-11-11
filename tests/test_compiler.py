import numpy as np
import pytest
from qibo import Circuit, gates

from qibolab import create_platform
from qibolab.compilers.compiler import Compiler


@pytest.fixture
def dummy():
    return create_platform("dummy")


@pytest.fixture
def dummy_couplers():
    return create_platform("dummy_couplers")


@pytest.fixture
def compiler():
    return Compiler.default()


def test_measurement_timings(dummy, compiler):
    q0, q1 = 0, 1
    circ = Circuit(2)
    circ.add(gates.GPI2(q0, phi=0.0))
    circ.add(gates.GPI2(q1, phi=np.pi / 2))
    circ.add(gates.GPI2(q1, phi=np.pi))
    
    # put measurement in different moments
    circ.add(gates.M(q0))
    circ.add(gates.M(q1))
    
    # make sure they are in different moments before proceeding
    assert not any([len([gate for gate in m if isinstance(gate, gates.M)]) == 2 for m in circ.queue.moments])
    
    circ._wire_names = [q0, q1]
    sequence, _ = compiler.compile(circ, dummy)
    
    MEASUREMENT_DURATION = 2000
    for pulse in sequence.ro_pulses:
        # assert that measurements don't happen one after another
        assert not pulse.start >= MEASUREMENT_DURATION
    

def test_coupler_pulse_timing(dummy_couplers, compiler):
    q0, q1, q2 = 0, 1, 2
    circ = Circuit(3)
    circ.add(gates.GPI2(q0, phi=0.0))
    circ.add(gates.GPI2(q0, phi=np.pi))
    circ.add(gates.GPI2(q1, phi=0.0))
    circ.add(gates.CZ(q1, q2))
    
    circ._wire_names = [q0, q1, q2]
    sequence, _ = compiler.compile(circ, dummy_couplers)
    
    coupler_pulse = sequence.cf_pulses[0]
    assert coupler_pulse.start == 40
