import pytest

from qibolab import Platform


def test_sequence(platform: Platform):
    q0 = platform.natives.single_qubit[0]
    mz = q0.MZ()
    sequences = [(mz, 0), (q0.RX() | mz, 1), (q0.RX() | q0.RX() | mz, 0)]
    for ps, mean in sequences:
        res = platform.execute([ps], nshots=1e4)[mz[0][1].id]
        assert pytest.approx(res.mean(), abs=1e-1) == mean
