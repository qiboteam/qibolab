from qibolab import create_platform
from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.pulses import Delay


def test_sequence_creation():
    platform = create_platform("dummy")

    single = platform.natives.single_qubit
    two = platform.natives.two_qubit

    # How a complex sequence is supposed to be constructed
    # ----------------------------------------------------

    p02 = two[(0, 2)]
    p12 = two[(1, 2)]
    q0 = single[0]
    q1 = single[1]
    q2 = single[2]
    ch1 = platform.qubits[1]

    seq = (
        q1.RX()
        | p12.CZ()
        | [(ch1.drive, Delay(duration=6.5))]
        | q2.RX()
        | q0.RX12()
        | p02.CZ()
    )
    for q in range(3):
        seq |= single[q].MZ()

    # ----------------------------------------------------

    nshots = 17
    res = platform.execute([seq], ExecutionParameters(nshots=nshots))

    for r in res.values():
        assert r.shape == (nshots,)
