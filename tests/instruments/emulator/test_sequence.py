import pytest

from qibolab import Platform


def test_ground_state(platform: Platform):
    q0 = platform.natives.single_qubit[0]
    seq = q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=1e-1) == 0
