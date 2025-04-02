"""Testing basic pulse sequences with emulator platforms."""

import pytest

from qibolab import Platform


def test_ground_state(platform: Platform):
    q0 = platform.natives.single_qubit[0]
    seq = q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=1e-1) == 0


def test_superposition_state(platform: Platform):
    q0 = platform.natives.single_qubit[0]
    seq = q0.RX90() | q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=1e-1) == 0.5


def test_excited_state(platform: Platform):
    q0 = platform.natives.single_qubit[0]
    seq = q0.RX() | q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=1e-1) == 1


def test_second_excited_state(platform: Platform):
    q0 = platform.natives.single_qubit[0]
    if q0.RX12 is None:
        pytest.skip(f"Skipping due to missing RX12 for platform {platform}.")
    seq = q0.RX() | q0.RX12() | q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=1e-1) == 2
