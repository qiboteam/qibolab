from qibolab import create_platform
from qibolab.execution_parameters import ExecutionParameters
from qibolab.pulses import Delay


def test_dummy_execute_pulse_sequence_couplers():
    platform = create_platform("dummy")

    single = platform.natives.single_qubit
    two = platform.natives.two_qubit

    # How a complex sequence is supposed to be constructed
    # ----------------------------------------------------

    p02 = two[(0, 2)]
    p12 = two[(1, 2)]
    q1 = single[1]
    q2 = single[2]

    seq = q1.RX() | p12.CZ() | [("", Delay())] | q2.RX() | p02.CZ()
    for q in range(3):
        seq |= single[q].MZ()

    # ----------------------------------------------------

    nshots = 17
    res = platform.execute([seq], ExecutionParameters(nshots=nshots))

    for r in res.values():
        assert r.shape == (nshots,)
