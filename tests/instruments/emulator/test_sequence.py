"""Testing basic pulse sequences with emulator platforms."""

import numpy as np
import pytest

from qibolab import Platform, Pulse, PulseSequence, Rectangular, VirtualZ


def test_ground_state(platform: Platform):
    """Test the ground state of a qubit with emulator."""
    q0 = platform.natives.single_qubit[0]
    seq = q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=1e-2) == 0


def test_superposition_state(platform: Platform):
    """Test superposition state of a qubit with emulator."""
    q0 = platform.natives.single_qubit[0]
    seq = q0.RX90() | q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=5e-2) == 0.5


def test_excited_state(platform: Platform):
    """Test excited state of a qubit with emulator."""
    q0 = platform.natives.single_qubit[0]
    seq = q0.RX() | q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=5e-2) == 1


def test_second_excited_state(platform: Platform):
    """Test second excited state of a qubit with emulator."""
    q0 = platform.natives.single_qubit[0]
    if q0.RX12 is None:
        pytest.skip(f"Skipping due to missing RX12 for platform {platform}.")
    seq = q0.RX() | q0.RX12() | q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=1e-1) == 2


def test_virtualz_sequence(platform: Platform):
    """Test virtual Z sequence of a qubit with emulator."""
    q0 = platform.natives.single_qubit[0]
    ch = platform.qubits[0].drive
    seq = (
        q0.RX90()
        + PulseSequence([(ch, VirtualZ(phase=-np.pi / 2))])
        + q0.R(theta=np.pi / 2, phi=np.pi / 2)
    )
    seq |= q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=5e-2) == 1


def test_detuning_flux_pulse(platform: Platform):
    """Test detuning caused by flux pulse."""
    q0 = platform.natives.single_qubit[0]
    flux_channel = platform.qubits[0].flux

    # move the qubit away while applying RX pulse
    seq = PulseSequence(
        [
            (flux_channel, Pulse(duration=20, amplitude=0.5, envelope=Rectangular())),
        ]
    )
    seq += q0.RX()
    seq |= q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=5e-2) != 1


def test_detuning_static_bias(platform: Platform):
    """Test detuning caused by static bias."""
    q0 = platform.natives.single_qubit[0]
    flux_channel = platform.qubits[0].flux
    seq = q0.RX() | q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    updates = [{flux_channel: {"offset": 0.5}}]
    res = platform.execute([seq], nshots=1e4, updates=updates)[acq_handle]
    assert pytest.approx(res.mean(), abs=5e-2) != 1


def test_detuning_second_excited_state(platform: Platform):
    """Test detuning second excited state of a qubit with emulator."""
    q0 = platform.natives.single_qubit[0]
    if q0.RX12 is None:
        pytest.skip(f"Skipping due to missing RX12 for platform {platform}.")
    flux_channel = platform.qubits[0].flux
    seq = q0.RX()
    seq |= [
        (flux_channel, Pulse(duration=20, amplitude=0.5, envelope=Rectangular()))
    ] + q0.RX12()
    seq |= q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    # since we apply a flux pulse the qubit stays at 1
    assert pytest.approx(res.mean(), abs=1e-1) == 1
